"""Telegram notification sender.

Sends trading signals, order fills, daily reports, and error alerts
to a Telegram chat via the Bot API. Uses plain HTTP requests to
avoid heavy dependencies.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import structlog

from tradingbot.config import EnvSettings

logger = structlog.get_logger()

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends notifications to Telegram."""

    def __init__(self, env: EnvSettings | None = None):
        env = env or EnvSettings()
        self._token = env.telegram_bot_token
        self._chat_id = env.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.warning("telegram_not_configured")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _send(self, text: str) -> bool:
        """Send a message via Telegram Bot API using urllib (no extra deps)."""
        if not self._enabled:
            return False

        import urllib.request
        import urllib.error

        url = TELEGRAM_API_URL.format(token=self._token)
        params = f"chat_id={self._chat_id}&text={quote(text)}&parse_mode=HTML"
        full_url = f"{url}?{params}"

        try:
            # Run blocking HTTP request in executor to not block event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(full_url, timeout=10),
            )
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.error("telegram_send_error", error=str(e))
            return False

    async def send_signal(self, message: str) -> bool:
        """Send a trading signal notification."""
        text = f"📊 <b>Signal</b>\n{message}"
        return await self._send(text)

    async def send_fill(self, message: str) -> bool:
        """Send an order fill notification."""
        text = f"✅ <b>Order Filled</b>\n{message}"
        return await self._send(text)

    async def send_error(self, message: str) -> bool:
        """Send an error alert."""
        text = f"🚨 <b>Error</b>\n{message}"
        return await self._send(text)

    async def send_daily_report(self, report: str) -> bool:
        """Send daily performance report."""
        text = f"📈 <b>Daily Report</b>\n{report}"
        return await self._send(text)
