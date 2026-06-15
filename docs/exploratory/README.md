# Exploratory / Planned Features

These modules are **prototypes for planned v3 features**. They are **not wired into**
the live trading or training pipeline and are kept here for reference. None of them are
imported by `algo_v2.live.execute`, `algo_v2.training.train_ensemble`, or
`algo_v2.evaluation.validate`.

Each requires extra packages that are intentionally **not** in the default install
(to keep the production install lean). They are available via the `exploratory` extra
— `pip install -e ".[exploratory]"` — or install them manually before integrating any
of these.

| File | Purpose | Extra dependencies |
|---|---|---|
| `portfolio_optimizer.py` | Mean-variance volatility guard (`SafetyOptimizer`) that vetoes/rebalances ensemble allocations exceeding a max-volatility limit. | `pyportfolioopt`, `cvxpy`, `osqp`, `clarabel` |
| `xai_explainer.py` | SHAP-based trade explanations (`XAIExplainer`) that rank which input features drove a bot's decision. | `shap` |
| `sentiment_scraper.py` | FinBERT news-sentiment scraper that writes a per-ticker daily sentiment score. | `transformers`, `sentencepiece`, `torch` |

## Notes before integrating

- `portfolio_optimizer.py` references "35 bots" in its docstring — the ensemble is now
  **42 bots**. Update the comment and confirm the weights dict it receives matches the
  current live-executor allocation format before wiring it in.
- `sentiment_scraper.py` hardcodes `DB_PATH = 'sentiment_db.csv'` and loads FinBERT at
  import time. Move the path to `config.py` and guard the model load behind a function
  before production use. The live pipeline currently sets `sentiment = 0.0` (neutral);
  this scraper would replace that.
- `xai_explainer.py` needs the caller to supply a `model_predict_fn` wrapper around an
  RLlib policy and a background-state sample.
