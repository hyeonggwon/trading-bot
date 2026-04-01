"""Core backtesting engine.

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

# Map timeframe strings to number of periods per year for annualization
PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 525_960,
    "3m": 175_320,
    "5m": 105_192,
    "15m": 35_064,
    "30m": 17_532,
    "1h": 8_766,
    "4h": 2_191.5,
    "1d": 365.25,
    "1w": 52.18,
}


class BacktestEngine:
    """Event-loop style backtesting engine.

    For each candle i (i >= 1):
        1. Build visible_df = candles[0..i-1]  (strategy sees only past)
        2. Compute indicators on visible_df
        3. Strategy generates signals based on visible_df
        4. Use candle i for: stop loss checks, order fills, equity snapshot
    This ensures the strategy never sees the candle it trades on.
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
        # Track entry orders per symbol for correct trade pairing (Bug #4 fix)
        self._entry_orders: dict[str, Order] = {}

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestReport:
        """Run backtest on historical data."""
        initial_balance = self.cash
        self.risk_manager.peak_equity = initial_balance

        symbol = self.strategy.symbols[0]
        if symbol not in data:
            raise ValueError(f"No data provided for symbol {symbol}")

        df_full = data[symbol].copy()

        # Apply date filters
        if self.config.backtest.start_date:
            start = pd.Timestamp(self.config.backtest.start_date, tz="UTC")
            df_full = df_full[df_full.index >= start]
        if self.config.backtest.end_date:
            end = pd.Timestamp(self.config.backtest.end_date, tz="UTC")
            df_full = df_full[df_full.index <= end]

        if df_full.empty:
            logger.warning("no_data_in_range")
            return self._build_report(initial_balance)

        # Detect data gaps and warn
        from tradingbot.data.storage import detect_gaps
        gaps = detect_gaps(df_full, self.strategy.timeframe)
        if gaps:
            logger.warning(
                "data_gaps_in_backtest",
                gaps=len(gaps),
                timeframe=self.strategy.timeframe,
            )

        logger.info(
            "backtest_start",
            symbol=symbol,
            candles=len(df_full),
            start=str(df_full.index[0]),
            end=str(df_full.index[-1]),
        )

        # Snapshot original columns to detect indicator leakage into df_full
        original_columns = set(df_full.columns)

        # Main loop: strategy sees [0..i-1], fills happen on candle i
        for i in range(1, len(df_full)):
            # CRITICAL: .copy() isolates visible_df so indicator mutations
            # do not leak back into df_full. Removing .copy() breaks anti-lookahead.
            visible_df = df_full.iloc[:i].copy()

            # Candle i is used for fills and stop loss checks
            fill_row = df_full.iloc[i]
            fill_candle = Candle(
                timestamp=df_full.index[i].to_pydatetime(),
                open=float(fill_row["open"]),
                high=float(fill_row["high"]),
                low=float(fill_row["low"]),
                close=float(fill_row["close"]),
                volume=float(fill_row["volume"]),
            )

            # Step 1: Check stop losses on fill candle
            stop_loss_fired = self._check_stop_losses(symbol, fill_candle)

            # Step 2: Fill pending orders
            self._process_pending_orders(fill_candle)

            # Step 3: Compute indicators on visible (past-only) data
            visible_df = self.strategy.indicators(visible_df)

            # Verify indicator columns did not leak into df_full
            assert set(df_full.columns) == original_columns, (
                "Anti-lookahead violation: indicator columns leaked into source data"
            )

            # Step 4: Check exits for open positions
            if symbol in self.positions:
                exit_signal = self.strategy.should_exit(
                    visible_df, symbol, self.positions[symbol]
                )
                if exit_signal:
                    self._handle_signal(exit_signal, fill_candle)

            # Step 5: Check entries (Bug #5: no entry on stop-loss candle)
            if symbol not in self.positions and not stop_loss_fired:
                entry_signal = self.strategy.should_entry(visible_df, symbol)
                if entry_signal:
                    self._handle_signal(entry_signal, fill_candle)

            # Step 6: Record equity & update peak (Bug #9 fix)
            prices = {symbol: float(fill_row["close"])}
            equity = self._calculate_equity(prices)
            self.risk_manager.update_peak_equity(equity)
            self.equity_snapshots.append((fill_candle.timestamp, equity))

        # Close any remaining positions at last close
        if symbol in self.positions:
            last_row = df_full.iloc[-1]
            last_candle = Candle(
                timestamp=df_full.index[-1].to_pydatetime(),
                open=float(last_row["open"]),
                high=float(last_row["high"]),
                low=float(last_row["low"]),
                close=float(last_row["close"]),
                volume=float(last_row["volume"]),
            )
            self._force_close_position(symbol, last_candle)

        report = self._build_report(initial_balance)
        logger.info(
            "backtest_complete",
            trades=report.total_trades,
            total_return=f"{report.total_return:.2%}",
            sharpe=f"{report.sharpe_ratio:.2f}",
        )
        return report

    def _handle_signal(self, signal: Signal, fill_candle: Candle) -> None:
        """Process a signal: validate with risk manager, then execute.

        Uses fill_candle.open for all price-dependent calculations (equity,
        position sizing, stop loss) because at the moment of fill, only the
        open price is known — not the close.
        """
        # Use open price for valuation — close is not yet known at fill time
        fill_price_estimate = fill_candle.open
        prices = {signal.symbol: fill_price_estimate}

        portfolio = PortfolioState(
            timestamp=signal.timestamp,
            cash=self.cash,
            positions=list(self.positions.values()),
        )

        if not self.risk_manager.validate_signal(signal, portfolio, prices):
            return

        if signal.signal_type == SignalType.LONG_ENTRY:
            # Simulate fill first to get actual execution price
            order = Order(
                id=str(uuid.uuid4())[:8],
                symbol=signal.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0,  # placeholder, sized below
                created_at=signal.timestamp,
            )
            fill = self.simulator.simulate_fill(order, fill_candle)
            if not fill.filled:
                return

            # Size position and anchor stop loss to actual fill price
            equity = self._calculate_equity(prices)
            stop_loss = self.risk_manager.calculate_stop_loss(fill.fill_price)
            quantity = self.risk_manager.calculate_position_size(
                fill.fill_price, stop_loss, equity
            )
            if quantity <= 0:
                return

            order.quantity = quantity
            # Recalculate fee for actual quantity
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
        self,
        order: Order,
        fill_price: float,
        fee: float,
        timestamp: datetime,
        stop_loss: float,
    ) -> None:
        """Execute a buy order."""
        cost = fill_price * order.quantity + fee
        if cost > self.cash:
            # Not enough cash — reduce quantity and recalculate fee (Bug #3 fix)
            max_quantity = self.cash / (fill_price * (1 + self.config.backtest.fee_rate))
            order.quantity = max_quantity
            fee = fill_price * order.quantity * self.config.backtest.fee_rate
            cost = fill_price * order.quantity + fee

        if order.quantity <= 0:
            return

        self.cash = max(0.0, self.cash - cost)  # guard floating-point epsilon
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

        # Store entry order keyed by symbol (Bug #4 fix)
        self._entry_orders[order.symbol] = order

    def _execute_sell(
        self, order: Order, fill_price: float, fee: float, timestamp: datetime
    ) -> None:
        """Execute a sell order and record the trade."""
        if order.symbol not in self.positions:
            return

        revenue = fill_price * order.quantity - fee
        self.cash += revenue

        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.fee = fee

        # Pair with the correct entry order (Bug #4 fix)
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
        """Check and execute stop losses. Returns True if stop loss fired."""
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

    def _process_pending_orders(self, candle: Candle) -> None:
        """Process any pending limit orders."""
        remaining = []
        for order in self.pending_orders:
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
        """Force close a position at the end of backtest."""
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
        """Calculate total equity (cash + position values)."""
        position_value = sum(
            prices.get(p.symbol, p.entry_price) * p.size
            for p in self.positions.values()
        )
        return self.cash + position_value

    def _build_report(self, initial_balance: float) -> BacktestReport:
        """Build the final backtest report."""
        if self.equity_snapshots:
            equity_curve = pd.Series(
                [e for _, e in self.equity_snapshots],
                index=pd.DatetimeIndex([t for t, _ in self.equity_snapshots]),
                name="equity",
            )
        else:
            equity_curve = pd.Series(dtype=float, name="equity")

        # After force-close, all positions are liquidated → cash is the true balance.
        # Using equity snapshot would be wrong because force-close fills at open+slippage,
        # while the snapshot was computed at close price. (Bug F6 fix)
        final_balance = self.cash

        timeframe = self.strategy.timeframe
        return BacktestReport(
            trades=self.completed_trades,
            equity_curve=equity_curve,
            initial_balance=initial_balance,
            final_balance=final_balance,
            timeframe=timeframe,
        )
