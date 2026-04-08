"""Core backtesting engine.

Supports single-symbol and multi-symbol backtesting.
Iterates through historical candles chronologically. The strategy sees
candles [0..i-1] (all confirmed), then orders fill at candle i's open.
This eliminates lookahead bias: the decision is made before the fill bar.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd
import structlog

from tradingbot.backtest.report import BacktestReport
from tradingbot.backtest.simulator import OrderSimulator
from tradingbot.config import AppConfig
from tradingbot.core.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    SignalType,
)
from tradingbot.core.models import Candle, Order, PortfolioState, Position, Signal, Trade
from tradingbot.risk.manager import RiskManager
from tradingbot.strategy.base import Strategy

logger = structlog.get_logger()


class BacktestEngine:
    """Event-loop style backtesting engine supporting multiple symbols.

    For each timestamp across all symbols:
        1. Build visible_df per symbol = candles[0..i-1]
        2. Check stop losses and fill pending orders
        3. Compute indicators, check exits, check entries
        4. Record portfolio equity snapshot
    """

    def __init__(
        self,
        strategy: Strategy,
        config: AppConfig | None = None,
    ):
        self.strategy = strategy
        self.config = config or AppConfig()
        self.simulator = OrderSimulator(self.config.backtest)
        self.risk_manager = RiskManager(self.config.risk)

        # State
        self.cash: float = self.config.trading.initial_balance
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.pending_orders: list[Order] = []
        self.completed_trades: list[Trade] = []
        self.equity_snapshots: list[tuple[datetime, float]] = []
        self._entry_orders: dict[str, Order] = {}
        self._last_known_prices: dict[str, float] = {}

    def run(
        self,
        data: dict[str, pd.DataFrame],
        precomputed_indicators: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestReport:
        """Run backtest on historical data.

        Args:
            data: Dict mapping symbol to OHLCV DataFrame. Supports single
                  or multiple symbols.
        """
        # Reset state for safe reuse of engine instances
        initial_balance = self.config.trading.initial_balance
        self.cash = initial_balance
        self.positions.clear()
        self.pending_orders.clear()
        self.completed_trades.clear()
        self.equity_snapshots.clear()
        self._entry_orders.clear()
        self._last_known_prices.clear()
        self.risk_manager.peak_equity = initial_balance
        # Clear strategy-side caches (e.g., CombinedStrategy._entry_indices)
        if hasattr(self.strategy, "_entry_indices"):
            self.strategy._entry_indices.clear()

        symbols = self.strategy.symbols
        available_symbols = [s for s in symbols if s in data]
        if not available_symbols:
            raise ValueError(f"No data provided for symbols: {symbols}")

        # Prepare per-symbol data
        symbol_data: dict[str, pd.DataFrame] = {}
        for sym in available_symbols:
            df = data[sym].copy()
            # Deduplicate timestamps to prevent get_loc returning slice
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()
            if self.config.backtest.start_date:
                start = pd.Timestamp(self.config.backtest.start_date, tz="UTC")
                df = df[df.index >= start]
            if self.config.backtest.end_date:
                end = pd.Timestamp(self.config.backtest.end_date, tz="UTC")
                df = df[df.index <= end]
            if not df.empty:
                symbol_data[sym] = df

        if not symbol_data:
            logger.warning("no_data_in_range")
            return self._build_report(initial_balance)

        # Detect gaps per symbol
        from tradingbot.data.storage import detect_gaps
        for sym, df in symbol_data.items():
            gaps = detect_gaps(df, self.strategy.timeframe)
            if gaps:
                logger.warning("data_gaps_in_backtest", symbol=sym, gaps=len(gaps))

        # Build unified timeline from all symbols
        all_timestamps = sorted(
            set().union(*(df.index for df in symbol_data.values()))
        )

        logger.info(
            "backtest_start",
            symbols=list(symbol_data.keys()),
            total_timestamps=len(all_timestamps),
            start=str(all_timestamps[0]),
            end=str(all_timestamps[-1]),
        )

        # Pre-compute indicators on full DataFrame per symbol (O(N) total)
        # Safe for anti-lookahead: all indicators use only past data (rolling, shift, etc.)
        # Future access is prevented by slicing indicator_data[sym].iloc[:idx]
        # Strategies that depend on DataFrame length (e.g., resampling) opt out
        # via supports_precompute = False and use per-iteration computation.
        use_precompute = self.strategy.supports_precompute
        indicator_data: dict[str, pd.DataFrame] = {}
        if use_precompute:
            if precomputed_indicators:
                # Use externally pre-computed indicators (shared across strategies)
                indicator_data = precomputed_indicators
            else:
                for sym, df in symbol_data.items():
                    indicator_data[sym] = self.strategy.indicators(df.copy())
                    # Freeze to prevent accidental mutation — strategies only read
                    indicator_data[sym].values.flags.writeable = False

        # Track per-symbol candle index (how many candles have been seen)
        symbol_indices: dict[str, int] = {sym: 0 for sym in symbol_data}
        # Original columns for anti-lookahead assertion (per-iteration path only)
        if not use_precompute:
            original_columns: dict[str, set] = {
                sym: set(df.columns) for sym, df in symbol_data.items()
            }

        # Pre-build timestamp→index dicts and numpy arrays for fast lookup
        symbol_ts_to_idx: dict[str, dict] = {
            sym: {ts: i for i, ts in enumerate(df.index)}
            for sym, df in symbol_data.items()
        }
        # Extract OHLCV as numpy arrays — avoids pandas iloc overhead in hot loop
        ohlcv_arrays: dict[str, dict] = {
            sym: {
                "open": df["open"].values,
                "high": df["high"].values,
                "low": df["low"].values,
                "close": df["close"].values,
                "volume": df["volume"].values,
            }
            for sym, df in symbol_data.items()
        }

        # Main loop: iterate through unified timeline
        for ts in all_timestamps:
            stop_loss_fired_symbols: set[str] = set()
            fill_candles: dict[str, Candle] = {}

            # Phase 1: Update indices and process fills/stop losses per symbol
            for sym in list(symbol_data.keys()):
                idx = symbol_ts_to_idx[sym].get(ts)
                if idx is None:
                    continue

                symbol_indices[sym] = idx

                if idx == 0:
                    continue  # Need at least 1 prior candle

                arrays = ohlcv_arrays[sym]
                fill_candle = Candle(
                    timestamp=ts.to_pydatetime(),
                    open=float(arrays["open"][idx]),
                    high=float(arrays["high"][idx]),
                    low=float(arrays["low"][idx]),
                    close=float(arrays["close"][idx]),
                    volume=float(arrays["volume"][idx]),
                )
                fill_candles[sym] = fill_candle

                # Stop losses
                if self._check_stop_losses(sym, fill_candle):
                    stop_loss_fired_symbols.add(sym)
                    # Clear strategy's entry_index cache for stopped-out positions
                    if hasattr(self.strategy, "_entry_indices"):
                        self.strategy._entry_indices.pop(sym, None)

                # Pending orders — only process orders for this symbol's candle
                self._process_pending_orders(fill_candle, sym)

            # Phase 2: Strategy evaluation per symbol
            for sym in list(symbol_data.keys()):
                if sym not in fill_candles:
                    continue

                idx = symbol_indices[sym]

                # Visible data: candles [0..idx-1] (anti-lookahead)
                if use_precompute:
                    # Zero-copy view — anti-lookahead enforced by slice bounds
                    visible_df = indicator_data[sym].iloc[:idx]
                else:
                    visible_df = symbol_data[sym].iloc[:idx].copy()
                    visible_df = self.strategy.indicators(visible_df)
                    # Verify no leakage in per-iteration path
                    assert set(symbol_data[sym].columns) == original_columns[sym], (
                        f"Anti-lookahead violation on {sym}: indicator columns leaked"
                    )

                fill_candle = fill_candles[sym]

                # Check exits
                if sym in self.positions:
                    exit_signal = self.strategy.should_exit(
                        visible_df, sym, self.positions[sym]
                    )
                    if exit_signal:
                        self._handle_signal(exit_signal, fill_candle)

                # Check entries (no entry if stop loss fired or already positioned)
                if sym not in self.positions and sym not in stop_loss_fired_symbols:
                    entry_signal = self.strategy.should_entry(visible_df, sym)
                    if entry_signal:
                        self._handle_signal(entry_signal, fill_candle)

            # Phase 3: Update last known prices and record portfolio equity
            for sym in symbol_data:
                idx = symbol_ts_to_idx[sym].get(ts)
                if idx is not None:
                    self._last_known_prices[sym] = float(ohlcv_arrays[sym]["close"][idx])

            equity = self._calculate_equity(self._last_known_prices)
            self.risk_manager.update_peak_equity(equity)
            self.equity_snapshots.append((ts.to_pydatetime(), equity))

        # Force close remaining positions
        for sym in list(self.positions.keys()):
            df = symbol_data.get(sym)
            if df is None or df.empty:
                continue
            last_row = df.iloc[-1]
            last_candle = Candle(
                timestamp=df.index[-1].to_pydatetime(),
                open=float(last_row["open"]),
                high=float(last_row["high"]),
                low=float(last_row["low"]),
                close=float(last_row["close"]),
                volume=float(last_row["volume"]),
            )
            self._force_close_position(sym, last_candle)

        report = self._build_report(initial_balance)
        logger.info(
            "backtest_complete",
            symbols=len(symbol_data),
            trades=report.total_trades,
            total_return=f"{report.total_return:.2%}",
            sharpe=f"{report.sharpe_ratio:.2f}",
        )
        return report

    def _handle_signal(self, signal: Signal, fill_candle: Candle) -> None:
        """Process a signal: validate with risk manager, then execute."""
        fill_price_estimate = fill_candle.open
        # Build prices dict with all current positions + signal symbol
        prices = {signal.symbol: fill_price_estimate}
        for pos in self.positions.values():
            if pos.symbol not in prices:
                prices[pos.symbol] = self._last_known_prices.get(pos.symbol, pos.entry_price)

        portfolio = PortfolioState(
            timestamp=signal.timestamp,
            cash=self.cash,
            positions=list(self.positions.values()),
        )

        if not self.risk_manager.validate_signal(signal, portfolio, prices):
            return

        if signal.signal_type == SignalType.LONG_ENTRY:
            order = Order(
                id=str(uuid.uuid4())[:8],
                symbol=signal.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0,
                created_at=signal.timestamp,
            )
            fill = self.simulator.simulate_fill(order, fill_candle)
            if not fill.filled:
                return

            equity = self._calculate_equity(prices)
            stop_loss = self.risk_manager.calculate_stop_loss(fill.fill_price)
            quantity = self.risk_manager.calculate_position_size(
                fill.fill_price, stop_loss, equity
            )
            quantity = quantity * signal.strength  # ML probability-based sizing
            if quantity <= 0:
                return

            order.quantity = quantity
            fee = fill.fill_price * quantity * self.config.backtest.fee_rate
            self._execute_buy(order, fill.fill_price, fee, fill_candle.timestamp, stop_loss)

        elif signal.signal_type == SignalType.LONG_EXIT:
            if signal.symbol in self.positions:
                pos = self.positions[signal.symbol]
                order = Order(
                    id=str(uuid.uuid4())[:8],
                    symbol=signal.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.size,
                    created_at=signal.timestamp,
                )
                fill = self.simulator.simulate_fill(order, fill_candle)
                if fill.filled:
                    self._execute_sell(order, fill.fill_price, fill.fee, fill_candle.timestamp)

    def _execute_buy(
        self, order: Order, fill_price: float, fee: float,
        timestamp: datetime, stop_loss: float,
    ) -> None:
        cost = fill_price * order.quantity + fee
        if cost > self.cash:
            max_quantity = self.cash / (fill_price * (1 + self.config.backtest.fee_rate))
            order.quantity = max_quantity
            fee = fill_price * order.quantity * self.config.backtest.fee_rate
            cost = fill_price * order.quantity + fee

        if order.quantity <= 0:
            return

        self.cash = max(0.0, self.cash - cost)
        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.fee = fee

        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            side=PositionSide.LONG,
            size=order.quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
        )
        self._entry_orders[order.symbol] = order

    def _execute_sell(
        self, order: Order, fill_price: float, fee: float, timestamp: datetime
    ) -> None:
        if order.symbol not in self.positions:
            return

        revenue = fill_price * order.quantity - fee
        self.cash += revenue

        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.fee = fee

        entry_order = self._entry_orders.pop(order.symbol, None)
        if entry_order is not None:
            trade = Trade(
                symbol=order.symbol,
                entry_order=entry_order,
                exit_order=order,
            )
            self.completed_trades.append(trade)

        del self.positions[order.symbol]

    def _check_stop_losses(self, symbol: str, candle: Candle) -> bool:
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        if pos.stop_loss is None:
            return False
        result = self.simulator.check_stop_loss(pos.stop_loss, candle, pos.size)
        if result is not None and result.filled:
            order = Order(
                id=str(uuid.uuid4())[:8],
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=pos.size,
                created_at=candle.timestamp,
            )
            self._execute_sell(order, result.fill_price, result.fee, candle.timestamp)
            logger.debug("stop_loss_triggered", symbol=symbol, price=result.fill_price)
            return True
        return False

    def _process_pending_orders(self, candle: Candle, symbol: str | None = None) -> None:
        """Process pending orders. If symbol is given, only process that symbol's orders."""
        remaining = []
        for order in self.pending_orders:
            if symbol is not None and order.symbol != symbol:
                remaining.append(order)
                continue
            fill = self.simulator.simulate_fill(order, candle)
            if fill.filled:
                if order.side == OrderSide.BUY:
                    stop_loss = self.risk_manager.calculate_stop_loss(fill.fill_price)
                    self._execute_buy(
                        order, fill.fill_price, fill.fee, candle.timestamp, stop_loss
                    )
                else:
                    self._execute_sell(
                        order, fill.fill_price, fill.fee, candle.timestamp
                    )
            else:
                remaining.append(order)
        self.pending_orders = remaining

    def _force_close_position(self, symbol: str, candle: Candle) -> None:
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        order = Order(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.size,
            created_at=candle.timestamp,
        )
        fill = self.simulator.simulate_fill(order, candle)
        if fill.filled:
            self._execute_sell(order, fill.fill_price, fill.fee, candle.timestamp)

    def _calculate_equity(self, prices: dict[str, float]) -> float:
        position_value = sum(
            prices.get(p.symbol, p.entry_price) * p.size
            for p in self.positions.values()
        )
        return self.cash + position_value

    def _build_report(self, initial_balance: float) -> BacktestReport:
        if self.equity_snapshots:
            equity_curve = pd.Series(
                [e for _, e in self.equity_snapshots],
                index=pd.DatetimeIndex([t for t, _ in self.equity_snapshots]),
                name="equity",
            )
        else:
            equity_curve = pd.Series(dtype=float, name="equity")

        final_balance = self.cash
        timeframe = self.strategy.timeframe
        return BacktestReport(
            trades=self.completed_trades,
            equity_curve=equity_curve,
            initial_balance=initial_balance,
            final_balance=final_balance,
            timeframe=timeframe,
        )
