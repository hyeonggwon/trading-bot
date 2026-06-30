"""Live/paper trading engine.

Async polling loop that:
1. Fetches latest candles from the exchange
2. Detects when a new candle is confirmed (closed)
3. Runs the strategy on confirmed candles
4. Executes signals through the exchange (paper or real)
5. Manages positions and risk

Uses the same Strategy interface as the backtest engine.
"""

from __future__ import annotations

import asyncio
import signal as signal_module
from datetime import UTC, date, datetime

import pandas as pd
import structlog

from tradingbot.config import AppConfig
from tradingbot.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, SignalType
from tradingbot.core.models import Position, Signal
from tradingbot.exchange.base import BaseExchange
from tradingbot.live.state import StateManager
from tradingbot.risk.manager import RiskManager
from tradingbot.strategy.base import Strategy

logger = structlog.get_logger()

# WebSocket prices older than this (seconds) are treated as stale — the engine
# falls back to a fresh REST ticker rather than acting on a cached price.
WS_PRICE_MAX_AGE_SECONDS = 60.0
# Between candle polls, monitor prices / stop losses at this cadence (seconds)
# so stops are enforced in real time instead of only at candle close.
MONITOR_INTERVAL_SECONDS = 10.0

# Timeframe to seconds for polling interval
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class LiveEngine:
    """Async live/paper trading engine.

    Supports both paper and live modes. In live mode, integrates
    OrderManager for order lifecycle and TradeValidator for safety.
    """

    def __init__(
        self,
        strategy: Strategy,
        exchange: BaseExchange,
        config: AppConfig,
        state_manager: StateManager | None = None,
        notifier: object | None = None,
        order_manager: object | None = None,
        trade_validator: object | None = None,
        ws_client: object | None = None,
    ):
        self.strategy = strategy
        self.exchange = exchange
        self.config = config
        self.risk_manager = RiskManager(config.risk)
        self.state = state_manager or StateManager()
        self.notifier = notifier
        self.order_manager = order_manager
        self.trade_validator = trade_validator
        self.ws_client = ws_client  # UpbitWebSocketClient for real-time prices

        self._running = False
        # Per-symbol last confirmed candle timestamp
        self._last_candle_ts: dict[str, datetime] = {}

    async def run(self) -> None:
        """Start the trading loop. Supports multiple symbols."""
        self._running = True
        symbols = self.strategy.symbols
        timeframe = self.strategy.timeframe
        poll_seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal_module.SIGINT, signal_module.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        ws_mode = self.ws_client is not None
        logger.info(
            "live_engine_start",
            symbols=symbols,
            timeframe=timeframe,
            poll_interval=f"{poll_seconds}s",
            mode="paper" if hasattr(self.exchange, '_feed') else "live",
            websocket=ws_mode,
        )

        # Load persisted state + restore real-money safety rails
        self._restore_state()
        if self.state.positions:
            logger.info("restored_positions", count=len(self.state.positions))

        # Initial warmup per symbol (parallel)
        warmup_candles = 200
        warmup_tasks = [
            self.exchange.fetch_ohlcv(sym, timeframe, limit=warmup_candles)
            for sym in symbols
        ]
        warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
        for sym, result in zip(symbols, warmup_results):
            if isinstance(result, Exception):
                logger.warning("warmup_failed", symbol=sym, error=str(result))
                continue
            if len(result) >= 2:
                self._last_candle_ts[sym] = result.index[-2].to_pydatetime()
                logger.info("warmup_complete", symbol=sym, candles=len(result))

        # Start WebSocket in background if available (real-time price updates)
        ws_task = None
        if self.ws_client is not None:
            ws_task = asyncio.create_task(self.ws_client.run())
            ws_task.add_done_callback(self._on_ws_task_done)
            logger.info("ws_started", symbols=len(symbols))

        # Main loop (candle polling — WebSocket provides prices, REST provides candles)
        monitor_interval = min(MONITOR_INTERVAL_SECONDS, poll_seconds)
        while self._running:
            try:
                await self._tick_all(symbols, timeframe)
            except Exception as e:
                logger.error("tick_error", error=str(e), type=type(e).__name__)
                if self.notifier and hasattr(self.notifier, 'send_error'):
                    await self.notifier.send_error(f"Tick error: {e}")

            # Between candle polls, monitor prices + stop losses at a faster
            # cadence (also keeps shutdown responsive — checks _running each tick).
            remaining = poll_seconds
            while remaining > 0 and self._running:
                wait = min(remaining, monitor_interval)
                await asyncio.sleep(wait)
                remaining -= wait
                if self._running and remaining > 0:
                    try:
                        await self._monitor_prices(symbols)
                    except Exception as e:
                        logger.error(
                            "monitor_error", error=str(e), type=type(e).__name__
                        )

        # Shutdown
        logger.info("live_engine_stopping")
        if self.ws_client is not None:
            self.ws_client.stop()
        if ws_task is not None:
            ws_task.cancel()
            try:
                await ws_task
            except (asyncio.CancelledError, Exception):
                pass  # Don't let WS errors block shutdown
        self._persist_state()
        await self.exchange.close()
        logger.info("live_engine_stopped")

    async def _tick_all(self, symbols: list[str], timeframe: str) -> None:
        """Single iteration — fetch candles, use WS prices or fetch tickers."""
        # Fetch candles for all symbols
        ohlcv_tasks = [self.exchange.fetch_ohlcv(sym, timeframe, limit=200) for sym in symbols]
        ohlcv_results = await asyncio.gather(*ohlcv_tasks, return_exceptions=True)

        # Resolve current prices (fresh WS prices, else REST tickers)
        tickers = await self._resolve_tickers(symbols)

        # Update equity using pre-fetched tickers (avoids redundant API calls)
        equity = await self._calculate_equity(tickers)
        self.risk_manager.update_peak_equity(equity)
        self.state.record_equity(equity)

        # Process each symbol
        for sym, result in zip(symbols, ohlcv_results):
            if isinstance(result, Exception):
                logger.warning("fetch_error", symbol=sym, error=str(result))
                continue
            await self._tick_symbol(sym, result, tickers.get(sym))

        # Persist state after processing all symbols
        self._persist_state()

    async def _resolve_tickers(self, symbols: list[str]) -> dict[str, dict]:
        """Resolve current prices for all symbols.

        Prefers WebSocket prices received within ``WS_PRICE_MAX_AGE_SECONDS``;
        stale or absent WS prices fall back to a fresh REST ticker pull so the
        engine never acts on an indefinitely-cached price. Fresh WS prices are
        also synced into the paper-exchange cache for accurate fills.
        """
        ws_prices = (
            self.ws_client.fresh_prices(WS_PRICE_MAX_AGE_SECONDS)
            if self.ws_client is not None
            else {}
        )
        if ws_prices and hasattr(self.exchange, "update_prices"):
            self.exchange.update_prices(ws_prices)

        # Use fresh WS prices where available; REST-fetch only the symbols still
        # missing. A per-symbol fallback (not all-or-nothing) ensures a quiet
        # symbol that simply hasn't ticked within the staleness window still
        # gets a price, so its stop loss is never silently dropped from
        # monitoring just because another symbol ticked.
        tickers: dict[str, dict] = {
            sym: {"last": ws_prices[sym]} for sym in symbols if sym in ws_prices
        }
        missing = [sym for sym in symbols if sym not in tickers]
        if missing:
            ticker_results = await asyncio.gather(
                *(self.exchange.fetch_ticker(sym) for sym in missing),
                return_exceptions=True,
            )
            for sym, res in zip(missing, ticker_results):
                if isinstance(res, dict):  # skip fetch_ticker failures
                    tickers[sym] = res
        return tickers

    async def _monitor_prices(self, symbols: list[str]) -> None:
        """Between-candle monitor: refresh prices, update equity, enforce stops.

        Runs at ``MONITOR_INTERVAL_SECONDS`` so stop losses are checked against
        live prices instead of only when a new candle closes (which on a 4h
        timeframe would leave positions unguarded for hours).
        """
        tickers = await self._resolve_tickers(symbols)
        equity = await self._calculate_equity(tickers)
        self.risk_manager.update_peak_equity(equity)
        self.state.record_equity(equity)
        for sym in symbols:
            position = self.state.positions.get(sym)
            if position is None:
                continue
            ticker = tickers.get(sym)
            price = float(ticker["last"]) if ticker and ticker.get("last") else None
            if price is None:
                continue
            await self._maybe_stop_out(sym, position, price)
        self._persist_state()

    def _restore_state(self) -> None:
        """Load persisted state and restore real-money safety rails.

        peak_equity is the drawdown circuit-breaker baseline; the validator's
        daily PnL is the daily-loss limit's running total. Without restoring
        them a restart silently resets both.
        """
        self.state.load()
        if self.state.peak_equity:
            self.risk_manager.peak_equity = self.state.peak_equity
        if self.trade_validator is not None and hasattr(
            self.trade_validator, "restore_daily_state"
        ):
            reset_date = (
                date.fromisoformat(self.state.daily_reset_date)
                if self.state.daily_reset_date
                else None
            )
            self.trade_validator.restore_daily_state(self.state.daily_pnl, reset_date)

    def _persist_state(self) -> None:
        """Snapshot in-memory risk state into the state manager and persist.

        peak_equity (drawdown circuit breaker) and the validator's daily PnL
        must survive restarts — both are real-money safety rails.
        """
        self.state.peak_equity = self.risk_manager.peak_equity
        if self.trade_validator is not None and hasattr(
            self.trade_validator, "daily_state"
        ):
            daily_pnl, reset_date = self.trade_validator.daily_state()
            self.state.daily_pnl = daily_pnl
            self.state.daily_reset_date = (
                reset_date.isoformat() if reset_date else None
            )
        self.state.save()

    async def _tick_symbol(
        self, symbol: str, df: pd.DataFrame, ticker: dict | None = None
    ) -> None:
        """Process a single symbol's candle data."""
        if df.empty or len(df) < 2:
            return

        confirmed_df = df.iloc[:-1].copy()
        confirmed_ts = confirmed_df.index[-1].to_pydatetime()

        # Check if we've already processed this candle for this symbol
        last_ts = self._last_candle_ts.get(symbol)
        if last_ts is not None and confirmed_ts <= last_ts:
            return

        self._last_candle_ts[symbol] = confirmed_ts

        logger.debug(
            "new_candle",
            symbol=symbol,
            timestamp=str(confirmed_ts),
            close=f"{confirmed_df['close'].iloc[-1]:,.0f}",
        )

        # Compute indicators
        confirmed_df = self.strategy.indicators(confirmed_df)

        # Use pre-fetched ticker or fallback to incomplete candle close
        # (incomplete candle close is the best real-time estimate when ticker unavailable)
        ticker_price = ticker.get("last") if ticker else None
        current_price = float(ticker_price) if ticker_price else float(df["close"].iloc[-1])

        # Check stop loss first, then strategy exit signals
        position = self.state.positions.get(symbol)
        if position is not None:
            stopped = await self._maybe_stop_out(symbol, position, current_price)
            if not stopped:
                exit_signal = self.strategy.should_exit(confirmed_df, symbol, position)
                if exit_signal:
                    await self._handle_exit(exit_signal, symbol, position)

        # Check entry signals (only if no position in this symbol)
        if symbol not in self.state.positions:
            entry_signal = self.strategy.should_entry(confirmed_df, symbol)
            if entry_signal:
                await self._handle_entry(entry_signal, symbol, current_price)

    async def _maybe_stop_out(
        self, symbol: str, position: Position, current_price: float
    ) -> bool:
        """Close the position if ``current_price`` breached its stop loss.

        Returns True if the stop fired and the position was closed. Shared by
        the candle path (``_tick_symbol``) and the between-candle monitor so
        stops are enforced in real time, not only at candle close.
        """
        if not position.stop_loss or current_price > position.stop_loss:
            return False

        logger.info(
            "stop_loss_triggered",
            symbol=symbol,
            current_price=f"{current_price:,.0f}",
            stop_loss=f"{position.stop_loss:,.0f}",
        )
        stop_signal = Signal(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            signal_type=SignalType.LONG_EXIT,
            price=current_price,
            strength=1.0,
        )
        await self._handle_exit(stop_signal, symbol, position)
        if symbol not in self.state.positions:
            if self.notifier and hasattr(self.notifier, 'send_signal'):
                await self.notifier.send_signal(
                    f"STOP LOSS {symbol}: price={current_price:,.0f}, "
                    f"stop={position.stop_loss:,.0f}"
                )
            return True
        logger.error("stop_loss_exit_failed", symbol=symbol)
        return False

    async def _handle_entry(
        self, signal_obj: Signal, symbol: str, current_price: float
    ) -> None:
        """Process an entry signal."""
        balance = await self.exchange.get_balance()
        cash = balance.get("KRW", 0)
        equity = await self._calculate_equity(balance=balance)

        # Validate with risk manager using actual cash balance
        from tradingbot.core.models import PortfolioState
        portfolio = PortfolioState(
            timestamp=datetime.now(UTC),
            cash=cash,
            positions=list(self.state.positions.values()),
        )
        prices = {symbol: current_price}
        if not self.risk_manager.validate_signal(signal_obj, portfolio, prices):
            logger.info("signal_rejected_by_risk_manager", symbol=symbol)
            return

        # Estimate fill price with slippage for conservative sizing
        slippage_pct = getattr(self.exchange, '_slippage_pct', 0.001)
        expected_price = current_price * (1 + slippage_pct)

        # Calculate position size using expected fill price
        stop_loss = self.risk_manager.calculate_stop_loss(expected_price)
        quantity = self.risk_manager.calculate_position_size(
            expected_price, stop_loss, equity
        )
        quantity = quantity * signal_obj.strength  # ML probability-based sizing (matches backtest)
        if quantity <= 0:
            return

        # Pre-trade validation with expected fill price
        if self.trade_validator is not None:
            if not self.trade_validator.validate_all(quantity, expected_price):
                logger.info("signal_rejected_by_validator", symbol=symbol)
                return

        # Execute order (via OrderManager if available, else direct)
        if self.order_manager is not None:
            order = await self.order_manager.submit_and_wait(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
        else:
            order = await self.exchange.create_order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )

        if order.status == OrderStatus.FILLED:
            if self.trade_validator is not None:
                self.trade_validator.record_order()

            actual_price = order.filled_price or current_price
            actual_stop_loss = self.risk_manager.calculate_stop_loss(actual_price)

            self.state.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.LONG,
                size=order.quantity,
                entry_price=actual_price,
                entry_time=datetime.now(UTC),
                stop_loss=actual_stop_loss,
            )
            # Track entry fee for accurate PnL on exit
            self.state.entry_fees[symbol] = order.fee or 0
            logger.info(
                "position_opened",
                symbol=symbol,
                quantity=f"{order.quantity:.8f}",
                price=f"{order.filled_price:,.0f}" if order.filled_price else "N/A",
            )
            if self.notifier and hasattr(self.notifier, 'send_signal'):
                await self.notifier.send_signal(
                    f"BUY {symbol}: qty={order.quantity:.8f}, "
                    f"price={order.filled_price:,.0f}"
                )

    async def _handle_exit(
        self, signal_obj: Signal, symbol: str, position: Position
    ) -> None:
        """Process an exit signal."""
        if self.order_manager is not None:
            order = await self.order_manager.submit_and_wait(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.size,
            )
        else:
            order = await self.exchange.create_order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.size,
            )

        if order.status == OrderStatus.FILLED:
            if self.trade_validator is not None:
                self.trade_validator.record_order()

            fill_price = order.filled_price or 0
            exit_fee = order.fee or 0
            entry_fee = self.state.entry_fees.pop(symbol, 0)
            pnl = (fill_price - position.entry_price) * position.size - entry_fee - exit_fee

            # Track PnL for daily loss limit
            if self.trade_validator is not None:
                self.trade_validator.record_trade_pnl(pnl)

            del self.state.positions[symbol]
            logger.info(
                "position_closed",
                symbol=symbol,
                entry=f"{position.entry_price:,.0f}",
                exit=f"{fill_price:,.0f}",
                pnl=f"{pnl:,.0f}",
            )
            if self.notifier and hasattr(self.notifier, 'send_signal'):
                await self.notifier.send_signal(
                    f"SELL {symbol}: price={fill_price:,.0f}, PnL={pnl:,.0f} KRW"
                )

    async def _calculate_equity(
        self,
        cached_tickers: dict | None = None,
        balance: dict | None = None,
    ) -> float:
        """Calculate total equity from exchange balances.

        Uses cached_tickers/balance if provided to avoid redundant API calls.
        """
        if balance is None:
            balance = await self.exchange.get_balance()
        equity = balance.get("KRW", 0)
        for currency, qty in balance.items():
            if currency == "KRW":
                continue
            symbol = f"{currency}/KRW"
            # Use cached ticker if available, otherwise fetch
            ticker = (cached_tickers or {}).get(symbol)
            if ticker:
                price = ticker.get("last")
                if price:
                    equity += float(price) * qty
            else:
                try:
                    fetched = await self.exchange.fetch_ticker(symbol)
                    equity += fetched["last"] * qty
                except Exception:
                    pass
        return equity

    @staticmethod
    def _on_ws_task_done(task: asyncio.Task) -> None:
        """Log unexpected WebSocket task failures."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("ws_task_failed", error=str(exc), type=type(exc).__name__)

    def _request_stop(self) -> None:
        """Handle shutdown signal."""
        logger.info("shutdown_requested")
        self._running = False
