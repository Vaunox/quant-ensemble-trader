import asyncio
import logging
import os
import sqlite3

import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from algo_v2.config import DB_PATH
from algo_v2.platform.notifications import send_telegram
from algo_v2.live.regime import RegimeDetector

logger = logging.getLogger(__name__)

PROBATION_DAYS = 90
SHARPE_LOOKBACK_DAYS = 90  # only recent history used for allocation weights


class MetaController:
    """
    Bandit-over-RL Meta-Controller.

    Routes capital to winning bots using Sharpe-weighted allocation (approximating UCB),
    stores per-bot performance history in SQLite, and triggers Telegram alerts when a
    bot underperforms for PROBATION_DAYS consecutive days.
    """

    def __init__(self):
        self._init_db()
        self.regime_detector = RegimeDetector()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_daily_performance (
                    date TEXT, bot_id TEXT, regime TEXT,
                    daily_return REAL, alpha REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_status (
                    bot_id TEXT PRIMARY KEY,
                    status TEXT DEFAULT "ACTIVE",
                    consecutive_losing_days INTEGER DEFAULT 0
                )
            """)

    def update_bot_performance(
        self, date: str, bot_id: str, daily_return: float, benchmark_return: float
    ) -> str | None:
        regime = self.regime_detector.detect_current_regime()
        alpha = daily_return - benchmark_return

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO bot_daily_performance (date, bot_id, regime, daily_return, alpha) VALUES (?,?,?,?,?)",
                (date, bot_id, regime, daily_return, alpha),
            )

            row = conn.execute(
                "SELECT status, consecutive_losing_days FROM bot_status WHERE bot_id = ?", (bot_id,)
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO bot_status (bot_id, status, consecutive_losing_days) VALUES (?,?,?)",
                    (bot_id, "ACTIVE", 0),
                )
                status, losing_days = "ACTIVE", 0
            else:
                status, losing_days = row

            losing_days = losing_days + 1 if alpha < 0 else 0

            alert_msg = None
            if losing_days >= PROBATION_DAYS and status == "ACTIVE":
                status = "WARNING"
                alert_msg = f"[WARNING] {bot_id} has negative alpha for {PROBATION_DAYS} days. /retrain {bot_id} or /ignore {bot_id}"
            elif losing_days >= PROBATION_DAYS * 2 and status == "RETRAINING":
                status = "CRITICAL"
                alert_msg = f"[CRITICAL] {bot_id} failed probation again. /pause {bot_id} to quarantine."

            conn.execute(
                "UPDATE bot_status SET status = ?, consecutive_losing_days = ? WHERE bot_id = ?",
                (status, losing_days, bot_id),
            )

        if alert_msg:
            send_telegram(alert_msg, parse_mode="HTML")
        return alert_msg

    def get_capital_allocation(self) -> dict:
        """Allocates capital proportional to Sharpe ratio in the current market regime."""
        regime = self.regime_detector.detect_current_regime()

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT bot_id, daily_return FROM bot_daily_performance WHERE regime = ? AND date >= date('now', ?)",
                conn,
                params=(regime, f"-{SHARPE_LOOKBACK_DAYS} days"),
            )
            status_df = pd.read_sql_query("SELECT bot_id, status FROM bot_status", conn)

        if df.empty:
            return {}

        stats = df.groupby("bot_id")["daily_return"].agg(["mean", "std"]).fillna(0)
        stats["sharpe"] = stats["mean"] / (stats["std"] + 1e-8)
        stats = stats.reset_index().merge(status_df, on="bot_id", how="left")
        stats = stats[stats["status"] != "PAUSED"]

        if stats.empty:
            return {}

        stats["weight"] = np.maximum(0, stats["sharpe"])
        total = stats["weight"].sum()
        stats["weight"] = stats["weight"] / total if total > 0 else 1.0 / len(stats)

        return dict(zip(stats["bot_id"], stats["weight"]))


async def _update_status(bot_id: str, status: str, reset_days: bool = False):
    with sqlite3.connect(DB_PATH) as conn:
        if reset_days:
            conn.execute(
                "UPDATE bot_status SET status = ?, consecutive_losing_days = 0 WHERE bot_id = ?",
                (status, bot_id),
            )
        else:
            conn.execute("UPDATE bot_status SET status = ? WHERE bot_id = ?", (status, bot_id))


async def handle_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_id = context.args[0] if context.args else "UNKNOWN"
    await _update_status(bot_id, "RETRAINING", reset_days=True)
    await update.message.reply_text(f"Retraining initiated for {bot_id}.")


async def handle_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_id = context.args[0] if context.args else "UNKNOWN"
    await _update_status(bot_id, "PAUSED")
    await update.message.reply_text(f"{bot_id} is PAUSED. Capital allocation set to 0%.")


async def handle_ignore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_id = context.args[0] if context.args else "UNKNOWN"
    await _update_status(bot_id, "ACTIVE", reset_days=True)
    await update.message.reply_text(f"Warning cleared for {bot_id}. 90-day grace period reset.")


async def start_telegram_listener():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — listener disabled.")
        return
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("retrain", handle_retrain))
    app.add_handler(CommandHandler("pause", handle_pause))
    app.add_handler(CommandHandler("ignore", handle_ignore))
    logger.info("Listening for probation commands...")
    await app.run_polling()


if __name__ == "__main__":
    import sys
    from algo_v2.platform.logging_config import configure_logging

    configure_logging("meta_controller")
    if len(sys.argv) > 1 and sys.argv[1] == "listen":
        asyncio.run(start_telegram_listener())
    else:
        mc = MetaController()
        logger.info("MetaController initialized. Run `python meta_controller.py listen` to start Telegram bot.")
