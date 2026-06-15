import os

# Project root = three levels up from this file (src/algo_v2/config.py -> repo root).
# Runtime artifacts (checkpoints, trained models) live at the project root and are
# anchored here so paths resolve correctly no matter where the package is imported from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")
EXPERT_DATA_DIR = os.path.join(PROJECT_ROOT, "expert_data")
LIVE_STATES_DIR = os.path.join(PROJECT_ROOT, "live_states")

# AWS region for CloudWatch metrics/logs (override with the AWS_REGION env var).
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

TICKER_LIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "BHARTIARTL.NS",
    "SBIN.NS",
    "INFY.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "LT.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "M&M.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "ASIANPAINT.NS",
    "COALINDIA.NS",
    "BAJAJFINSV.NS",
    "ADANIENT.NS",
    "NESTLEIND.NS",
    "JSWSTEEL.NS",
    "GRASIM.NS",
    "WIPRO.NS",
    "INDUSINDBK.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "TATASTEEL.NS",
    "TATACONSUM.NS",
    "APOLLOHOSP.NS",
    "HINDALCO.NS",
    "TECHM.NS",
    "EICHERMOT.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS",
    "UPL.NS",
    "BPCL.NS",
    "HDFCAMC.NS",
    "SHREECEM.NS",
]

# 12 technical indicators computed by FeatureEngineer + VIX + Sentiment = 14 total
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "wr_30",
    "atr_30",
    "mfi_30",
    "vwma_30",
    "vix",
    "sentiment",
]

# The 12 indicators FeatureEngineer computes (VIX and sentiment are added separately)
TECHNICAL_INDICATORS = INDICATORS[:12]

# Primary data file for all pipeline stages (GAN, expert, ensemble training).
# Switch this one constant to change the entire pipeline's data source.
# v1: indian_stocks_preprocessed.csv  (5yr,  2021-2026, 60k rows)
# v2: indian_stocks_15yr.csv          (16yr, 2010-2026, 198k rows) ← active
PRIMARY_DATA_PATH = "indian_stocks_15yr.csv"

# Raw-data fetch window for fetch_data_historical.py (regenerates PRIMARY_DATA_PATH).
# DATA_END_DATE = None → yfinance fetches through today.
DATA_START_DATE = "2010-01-01"
DATA_END_DATE = None

# Validation split for checkpoint selection (never used in training episodes).
# Walk-forward folds use real data 2022-2023; final test uses 2024-2026.
VAL_START_DATE = "2022-01-01"
VAL_END_DATE = "2024-01-01"
TEST_START_DATE = "2024-01-01"

HMAX = 5_000
INITIAL_CAPITAL = 1_000_000
TRADE_FEE_PCT = 0.0002
REWARD_SCALING = 1e-4
DOWNSIDE_PENALTY = 2.0


# ── Live trading ──────────────────────────────────────────────────────────────
ACTION_THRESHOLD = 0.3  # ensemble signal must exceed this magnitude to place an order

# ── MetaController persistence ────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_performance.db")

# ── Ensemble architecture registry ────────────────────────────────────────────
# Single source of truth for bot architecture definitions.
# train_ensemble, validate_ensemble, and live_executor all import from here
# so an architecture change only needs to be made in one place.
#
# ALGO_MAP (string → RLlib ConfigClass) is NOT here — importing Ray at config
# load time is expensive.  Each script builds its own ALGO_MAP after importing
# the relevant ConfigClasses.

ARCH_MAP = {
    "pure_mlp": {"custom_model": None, "use_lstm": False},
    "pure_lstm": {"custom_model": None, "use_lstm": True},
    "cnn_micro": {
        "custom_model": "dual_branch_cnn",
        "custom_model_config": {"use_micro": True, "use_macro": False, "use_attention": False},
        "use_lstm": False,
    },
    "cnn_macro": {
        "custom_model": "dual_branch_cnn",
        "custom_model_config": {"use_micro": False, "use_macro": True, "use_attention": False},
        "use_lstm": False,
    },
    "cnn_dual": {
        "custom_model": "dual_branch_cnn",
        "custom_model_config": {"use_micro": True, "use_macro": True, "use_attention": False},
        "use_lstm": False,
    },
    "transformer_cnn": {
        "custom_model": "dual_branch_cnn",
        "custom_model_config": {"use_micro": True, "use_macro": True, "use_attention": True},
        "use_lstm": False,
    },
    # custom_model_config intentionally omitted for god_tier: DualBranchCNNExtractor
    # defaults are all-True; passing it as **kwargs to LSTMWrapper.__init__() crashes.
    "god_tier": {"custom_model": "dual_branch_cnn", "use_lstm": True},
}

# SAC/TQC: off-policy replay destroys temporal order → LSTM learns nothing.
# Replace LSTM slots with feedforward temporal-attention equivalents.
SAC_TQC_ARCH_OVERRIDES = {
    "pure_lstm": {"custom_model": "indicator_attn", "use_lstm": False},
    "god_tier": {"custom_model": "temporal_attn", "use_lstm": False},
}

# MARWIL: offline transitions also destroy temporal order — same substitution.
MARWIL_ARCH_OVERRIDES = {
    "pure_lstm": {"custom_model": "indicator_attn", "use_lstm": False},
    "god_tier": {"custom_model": "temporal_attn", "use_lstm": False},
}


def make_env_kwargs(stock_dimension: int) -> dict:
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    return {
        "hmax": HMAX,
        "initial_amount": INITIAL_CAPITAL,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [TRADE_FEE_PCT] * stock_dimension,
        "sell_cost_pct": [TRADE_FEE_PCT] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": REWARD_SCALING,
        "downside_penalty": DOWNSIDE_PENALTY,
    }
