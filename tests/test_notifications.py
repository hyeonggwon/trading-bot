"""Tests for Telegram notification formatting."""

from __future__ import annotations

import pytest

from tradingbot.config import EnvSettings
from tradingbot.notifications.telegram import TelegramNotifier


@pytest.fixture
def notifier_and_sent(monkeypatch):
    """A configured notifier plus the list of texts handed to _send."""
    notifier = TelegramNotifier(
        EnvSettings(telegram_bot_token="token", telegram_chat_id="chat"),
    )
    sent: list[str] = []

    async def _capture(text: str) -> bool:
        sent.append(text)
        return True

    monkeypatch.setattr(notifier, "_send", _capture)
    return notifier, sent


class TestHtmlEscaping:
    """Messages go out with parse_mode=HTML, so raw angle brackets in the body
    make Telegram reject the request with a 400 — silently dropping exactly the
    error alerts an operator needs."""

    @pytest.mark.asyncio
    async def test_error_body_is_escaped(self, notifier_and_sent):
        notifier, sent = notifier_and_sent
        await notifier.send_error("Exit order failed: <Response [429]>")

        assert "&lt;Response [429]&gt;" in sent[0]
        assert "<Response" not in sent[0]
        assert "<b>Error</b>" in sent[0]  # our own markup survives

    @pytest.mark.asyncio
    async def test_signal_body_is_escaped(self, notifier_and_sent):
        notifier, sent = notifier_and_sent
        await notifier.send_signal("BUY A<B & C")

        assert "A&lt;B &amp; C" in sent[0]
        assert "<b>Signal</b>" in sent[0]
