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
    ):
        self.strategy = strategy
        self.exchange = exchange
        self.config = config
        self.risk_manager = RiskManager(config.risk)
        self.state = state_manager or StateManager()
        self.notifier = notifier
        self.order_manager = order_manager
        self.trade_validator = trade_validator

        self._running = False
        self._last_candle_ts: datetime | None = None

    async def run(self) -> None:
        """Start the trading loop."""
        self._running = True
        symbol = self.strategy.symbols[0]
        timeframe = self.strategy.timeframe
        poll_seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal_module.SIGINT, signal_module.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        logger.info(
            "live_engine_start",
            symbol=symbol,
            timeframe=timeframe,
            poll_interval=f"{poll_seconds}s",
            mode="paper" if hasattr(self.exchange, '_feed') else "live",
        )

        # Load persisted state
        self.state.load()
        if self.state.positions:
            logger.info("restored_positions", count=len(self.state.positions))

        # Initial data fetch for indicator warmup
        warmup_candles = 200
        df = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=warmup_candles)
        if len(df) >= 2:
            # Track last confirmed candle (exclude the possibly incomplete last one)
            self._last_candle_ts = df.index[-2].to_pydatetime()
            logger.info("warmup_complete", candles=len(df), last_confirmed=str(self._last_candle_ts))

        # Main loop
        while self._running:
            try:
                await self._tick(symbol, timeframe)
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
        self.state.save()
        await self.exchange.close()
        logger.info("live_engine_stopped")

    async def _tick(self, symbol: str, timeframe: str) -> None:
        """Single iteration of the trading loop."""
        # Fetch latest candles
        df = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        if df.empty:
            return

        # Use all candles except the last (which may be incomplete)
        if len(df) < 2:
            return
        confirmed_df = df.iloc[:-1].copy()

        # Track the last CONFIRMED candle timestamp, not the incomplete one
        confirmed_ts = confirmed_df.index[-1].to_pydatetime()
        if self._last_candle_ts is not None and confirmed_ts <= self._last_candle_ts:
            return  # No new confirmed candle yet

        self._last_candle_ts = confirmed_ts

        logger.debug(
            "new_candle",
            symbol=symbol,
            timestamp=str(confirmed_df.index[-1]),
            close=f"{confirmed_df['close'].iloc[-1]:,.0f}",
        )

        # Compute indicators
        confirmed_df = self.strategy.indicators(confirmed_df)

        # Fetch current price for equity/risk calculations
        ticker = await self.exchange.fetch_ticker(symbol)
        current_price = ticker["last"]

        # Update risk manager peak equity
        equity = await self._calculate_equity()
        self.risk_manager.update_peak_equity(equity)

        # Check exit signals for open positions
        position = self.state.positions.get(symbol)
        if position is not None:
            exit_signal = self.strategy.should_exit(confirmed_df, symbol, position)
            if exit_signal:
                await self._handle_exit(exit_signal, symbol, position)

        # Check entry signals (only if no position)
        if symbol not in self.state.positions:
            entry_signal = self.strategy.should_entry(confirmed_df, symbol)
            if entry_signal:
                await self._handle_entry(entry_signal, symbol, current_price)

        # Persist state
        self.state.save()

    async def _handle_entry(
        self, signal_obj: Signal, symbol: str, current_price: float
    ) -> None:
        """Process an entry signal."""
        equity = await self._calculate_equity()

        # Validate with risk manager
        from tradingbot.core.models import PortfolioState
        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            cash=0,  # Not directly applicable for live
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
            fee = order.fee or 0
            pnl = (fill_price - position.entry_price) * position.size - fee

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

    async def _calculate_equity(self) -> float:
        """Calculate total equity from exchange balances."""
        balance = await self.exchange.get_balance()
        equity = balance.get("KRW", 0)
        for currency, qty in balance.items():
            if currency == "KRW":
                continue
            # Try to get last price for this currency
            symbol = f"{currency}/KRW"
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                equity += ticker["last"] * qty
            except Exception:
                pass
        return equity

    def _request_stop(self) -> None:
        """Handle shutdown signal."""
        logger.info("shutdown_requested")
        self._running = False
