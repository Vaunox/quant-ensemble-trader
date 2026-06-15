"""
algo_v2.data.fetch — Regenerates the primary training dataset.

Downloads OHLCV for every ticker in TICKER_LIST over the configured data window,
adds technical indicators + VIX, fills the date×ticker grid, and writes the result
to PRIMARY_DATA_PATH (the file the entire pipeline reads).

Usage:
    python -m algo_v2.data.fetch
"""

import logging

import pandas as pd
import yfinance as yf

from algo_v2.config import (
    DATA_END_DATE,
    DATA_START_DATE,
    PRIMARY_DATA_PATH,
    TECHNICAL_INDICATORS,
    TICKER_LIST,
)
from algo_v2.platform.logging_config import configure_logging
from algo_v2.data.features import add_indicators, download_vix, grid_fill

logger = logging.getLogger(__name__)


def fetch_historical_data() -> pd.DataFrame:
    df_list = []
    for tic in TICKER_LIST:
        logger.info("Downloading %s ...", tic)
        try:
            raw = yf.download(tic, start=DATA_START_DATE, end=DATA_END_DATE, progress=False)
            if raw.empty:
                logger.warning("No data for %s — skipping", tic)
                continue
            raw = raw.reset_index()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower() for c in raw.columns]
            if "adj close" in raw.columns:
                raw = raw.drop(columns=["close"]).rename(columns={"adj close": "close"})
            raw["tic"] = tic
            df_list.append(raw)
        except Exception as exc:
            logger.error("Error downloading %s: %s", tic, exc)

    if not df_list:
        raise RuntimeError("No stock data downloaded.")

    df = pd.concat(df_list, ignore_index=True)
    df["date"] = df["date"].astype(str)

    tickers = sorted(df["tic"].unique().tolist())
    dates = sorted(df["date"].unique().tolist())
    df = grid_fill(df, dates, tickers)

    logger.info("Calculating technical indicators ...")
    processed = add_indicators(df, TECHNICAL_INDICATORS)
    processed = processed.ffill().bfill()

    logger.info("Downloading VIX ...")
    vix_df = download_vix(DATA_START_DATE, DATA_END_DATE)
    if vix_df is not None:
        processed = processed.merge(vix_df, on="date", how="left")
        processed["vix"] = processed["vix"].ffill().bfill()
    else:
        processed["vix"] = 0.0
        logger.warning("VIX unavailable — using 0.0")

    # Historical data has no real-time sentiment; initialize neutral.
    processed["sentiment"] = 0.0

    processed = processed.sort_values(["date", "tic"]).reset_index(drop=True)
    return processed


if __name__ == "__main__":
    configure_logging("fetch_data_historical")
    processed = fetch_historical_data()
    processed.to_csv(PRIMARY_DATA_PATH, index=False)
    logger.info("Saved %d rows to %s", len(processed), PRIMARY_DATA_PATH)
