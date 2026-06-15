"""Feature engineering and market-data helpers: grid fill, indicators, VIX."""

import pandas as pd
import yfinance as yf


def grid_fill(df: pd.DataFrame, dates, tickers) -> pd.DataFrame:
    """Ensures every (date, tic) pair exists, filling gaps with ffill then bfill."""
    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "tic"]).to_frame(index=False)
    df = pd.merge(grid, df, on=["date", "tic"], how="left")
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df = df.groupby("tic").ffill().bfill().reset_index(drop=True)
    df["tic"] = grid.sort_values(["tic", "date"])["tic"].values
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def add_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    """
    Calculate technical indicators for each stock using stockstats.
    df must have columns: date, tic, open, high, low, close, volume (lowercase).
    Processes each tic independently and returns the combined DataFrame.
    """
    import stockstats

    results = []
    for tic, group in df.groupby("tic"):
        g = group.sort_values("date").reset_index(drop=True).copy()
        try:
            ss = stockstats.StockDataFrame.retype(g[["open", "high", "low", "close", "volume"]].copy())
        except Exception:
            for ind in indicators:
                g[ind] = 0.0
            results.append(g)
            continue

        for ind in indicators:
            if ind == "vwma_30":
                denom = g["volume"].rolling(30, min_periods=1).sum()
                g["vwma_30"] = (g["close"] * g["volume"]).rolling(30, min_periods=1).sum() / denom
            else:
                try:
                    g[ind] = ss[ind].values
                except Exception:
                    g[ind] = 0.0

        results.append(g)

    return pd.concat(results, ignore_index=True)


def download_vix(start: str, end: str) -> pd.DataFrame | None:
    """Downloads India VIX and returns a (date, vix) DataFrame, or None on failure."""
    try:
        vix_df = yf.download("^INDIAVIX", start=start, end=end, progress=False)
        if vix_df.empty:
            return None
        vix_df = vix_df.reset_index()
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.droplevel(1)
        vix_df.columns = [c.lower() for c in vix_df.columns]
        vix_df = vix_df[["date", "close"]].rename(columns={"close": "vix"})
        vix_df["date"] = vix_df["date"].astype(str)
        return vix_df
    except Exception:
        return None
