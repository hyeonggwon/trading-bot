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
from datetime import datetime, timezone

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

        # Load persisted state
        self.state.load()
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
        while self._running:
            try:
                await self._tick_all(symbols, timeframe)
            except Exception as e:
                logger.error("tick_error", error=str(e), type=type(e).__name__)
                if self.notifier and hasattr(self.notifier, 'send_error'):
                    await self.notifier.send_error(f"Tick error: {e}")

            # Wait for next poll — check every 10s if we should stop
            remaining = poll_seconds
            while remaining > 0 and self._running:
                wait = min(remaining, 10)
                await asyncio.sleep(wait)
                remaining -= wait

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
        self.state.save()
        await self.exchange.close()
        logger.info("live_engine_stopped")

    async def _tick_all(self, symbols: list[str], timeframe: str) -> None:
        """Single iteration — fetch candles, use WS prices or fetch tickers."""
        # Fetch candles for all symbols
        ohlcv_tasks = [self.exchange.fetch_ohlcv(sym, timeframe, limit=200) for sym in symbols]
        ohlcv_results = await asyncio.gather(*ohlcv_tasks, return_exceptions=True)

        # Use WebSocket prices if available, otherwise fetch tickers via REST
        ws_prices = self.ws_client.last_prices if self.ws_client else {}
        if ws_prices:
            tickers = {
                sym: {"last": price}
                for sym, price in ws_prices.items()
            }
        else:
            ticker_tasks = [self.exchange.fetch_ticker(sym) for sym in symbols]
            ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
            tickers = {
                sym: res for sym, res in zip(symbols, ticker_results)
                if not isinstance(res, Exception)
            }

        # Update equity using pre-fetched tickers (avoids redundant API calls)
        equity = await self._calculate_equity(tickers)
        self.risk_manager.update_peak_equity(equity)

        # Process each symbol
        for sym, result in zip(symbols, ohlcv_results):
            if isinstance(result, Exception):
                logger.warning("fetch_error", symbol=sym, error=str(result))
                continue
            await self._tick_symbol(sym, result, tickers.get(sym))

        # Persist state after processing all symbols
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

        # Check exit signals for open positions
        position = self.state.positions.get(symbol)
        if position is not None:
            exit_signal = self.strategy.should_exit(confirmed_df, symbol, position)
            if exit_signal:
                await self._handle_exit(exit_signal, symbol, position)

        # Check entry signals (only if no position in this symbol)
        if symbol not in self.state.positions:
            entry_signal = self.strategy.should_entry(confirmed_df, symbol)
            if entry_signal:
                await self._handle_entry(entry_signal, symbol, current_price)

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
            timestamp=datetime.now(timezone.utc),
            cash=cash,
            positions=list(self.state.positions.values()),
        )
        prices = {symbol: current_price}
        if not self.risk_manager.validate_signal(signal_obj, portfolio, prices):
            logger.info("signal_rejected_by_risk_manager", symbol=symbol)
            return

        # Calculate position size
        stop_loss = self.risk_manager.calculate_stop_loss(current_price)
        quantity = self.risk_manager.calculate_position_size(
            current_price, stop_loss, equity
        )
        if quantity <= 0:
            return

        # Pre-trade validation (live mode safety)
        if self.trade_validator is not None:
            if not self.trade_validator.validate_all(quantity, current_price):
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

            self.state.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.LONG,
                size=order.quantity,
                entry_price=order.filled_price or current_price,
                entry_time=datetime.now(timezone.utc),
                stop_loss=stop_loss,
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
