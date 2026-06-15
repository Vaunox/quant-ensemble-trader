import logging
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Task 16: Hidden Markov Model (HMM) Regime Detection
    Downloads the NIFTY 50 Index (^NSEI) to calculate the macro-economic state of the Indian market.
    Classifies the current market into 3 Hidden States: Bull (0), Bear (1), Sideways (2).
    """

    _CACHE_TTL = 3600  # re-detect at most once per hour (live run calls this 43×)

    def __init__(self, start_date="2010-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date
        self.model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        self.trained = False
        self._cached_regime: str | None = None
        self._cache_time: float = 0.0

    def _fetch_nifty50(self):
        logger.debug("Fetching NIFTY 50 macro data...")
        df = yf.download("^NSEI", start=self.start_date, end=self.end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return df

    def _calculate_features(self, df):
        """Calculates Volatility and MACD for the HMM."""
        df["Returns"] = df["Close"].pct_change()
        df["Volatility"] = df["Returns"].rolling(window=10).std()

        # Simple MACD calculation
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA_12"] - df["EMA_26"]

        df.dropna(inplace=True)
        return df

    def train(self):
        """Trains the Gaussian HMM on historical NIFTY 50 data."""
        df = self._fetch_nifty50()
        df = self._calculate_features(df)

        # We feed Volatility and MACD into the HMM
        X = df[["Volatility", "MACD"]].values
        logger.info("[HMM] Training GaussianHMM on %d NIFTY 50 days...", len(X))
        self.model.fit(X)
        self.trained = True

        # Determine which hidden state corresponds to which regime
        # By looking at the mean volatility of each state
        means = self.model.means_
        # means[:, 0] is Volatility, means[:, 1] is MACD

        vols = means[:, 0]
        bear_state = np.argmax(vols)  # Bear market usually has the highest volatility
        bull_state = np.argmax(means[:, 1])  # Bull market has the highest MACD

        # The remaining state is sideways
        all_states = {0, 1, 2}
        all_states.discard(bear_state)
        all_states.discard(bull_state)

        if len(all_states) == 1:
            sideways_state = list(all_states)[0]
        else:
            # Fallback if argmax picked the same state (rare, but possible if MACD and Vol are highly correlated)
            sideways_state = list(all_states)[0]  # Just assign one

        self.state_map = {bull_state: "BULL", bear_state: "BEAR", sideways_state: "SIDEWAYS"}

        logger.info("[HMM] Regime mapping: %s", self.state_map)

    def detect_current_regime(self) -> str:
        """Predicts the regime for the most recent trading day.

        Result is cached for _CACHE_TTL seconds so repeated calls within
        a single live run (once per bot = 42 calls + 1 allocation call)
        only download NIFTY data and refit the HMM once.
        """
        now = time.time()
        if self._cached_regime and (now - self._cache_time) < self._CACHE_TTL:
            return self._cached_regime

        if not self.trained:
            self.train()

        df = self._fetch_nifty50()
        df = self._calculate_features(df)

        latest_X = df[["Volatility", "MACD"]].values[-1].reshape(1, -1)
        hidden_state = self.model.predict(latest_X)[0]
        regime = self.state_map.get(hidden_state, "UNKNOWN")

        self._cached_regime = regime
        self._cache_time = now
        return regime


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    detector = RegimeDetector()
    current_regime = detector.detect_current_regime()
    print(f"Current Indian Market Regime: {current_regime}")
