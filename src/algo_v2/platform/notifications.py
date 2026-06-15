"""
Telegram notification helpers.

All modules that need to send Telegram messages should import from here so
credentials are read from a single place and the HTTP retry logic is shared.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


def send_telegram(msg: str, parse_mode: str = "HTML") -> bool:
    """
    Send a Telegram message. Returns True on success, False on any failure.
    Silently skips (returns False) if credentials are not set.
    """
    token = os.environ.get(_TOKEN_ENV)
    chat_id = os.environ.get(_CHAT_ID_ENV)
    if not token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": parse_mode},
            timeout=10,
        )
        if r.status_code == 200:
            logger.debug("Telegram message sent")
            return True
        logger.warning("Telegram send failed: HTTP %s — %s", r.status_code, r.text[:120])
        return False
    except Exception as exc:
        logger.warning("Telegram error: %s", exc)
        return False
