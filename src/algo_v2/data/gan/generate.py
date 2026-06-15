import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from algo_v2.data.gan.train_gan import Generator, MODEL_PATH, PARAMS_PATH
from algo_v2.config import CHECKPOINTS_DIR, TECHNICAL_INDICATORS
from algo_v2.data.features import add_indicators

SYNTH_OUT = os.path.join(CHECKPOINTS_DIR, "phase2_synthetic", "indian_stocks_synthetic.csv")

LATENT_DIM = 100
SEQ_LEN = 24
GENERATION_DAYS = 15_000
BASE_PRICE = 1_000.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_market():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run gan_trainer.py first.")
        return
    if not os.path.exists(PARAMS_PATH):
        print(f"Error: {PARAMS_PATH} not found. Run gan_trainer.py first.")
        return

    print("Loading GAN and scaling parameters...")
    params = pd.read_csv(PARAMS_PATH, index_col=0)
    mean = params["mean"].values
    std = params["std"].values
    tickers = params.index.tolist()
    num_assets = len(tickers)

    netG = Generator(LATENT_DIM, 64, num_assets).to(device)
    netG.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    netG.eval()

    print(f"Generating {GENERATION_DAYS} days of synthetic returns for {num_assets} stocks...")
    batches = GENERATION_DAYS // SEQ_LEN + 1
    all_returns = []
    with torch.no_grad():
        for _ in range(batches):
            z = torch.randn(1, SEQ_LEN, LATENT_DIM).to(device)
            all_returns.append(netG(z).cpu().numpy()[0])

    synth_returns = (np.vstack(all_returns)[:GENERATION_DAYS] * std) + mean
    synth_close = BASE_PRICE * np.cumprod(1 + synth_returns, axis=0)

    print("Building synthetic market DataFrame...")
    start_date = datetime(1980, 1, 1)
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(GENERATION_DAYS)]

    rng = np.random.default_rng()
    records = []
    for t_idx, tic in enumerate(tickers):
        closes = synth_close[:, t_idx]
        for day in range(GENERATION_DAYS):
            c = closes[day]
            prev = closes[day - 1] if day > 0 else BASE_PRICE
            o = prev * (1 + rng.normal(0, 0.002))
            records.append(
                {
                    "date": dates[day],
                    "tic": tic,
                    "open": o,
                    "high": max(o, c) * (1 + abs(rng.normal(0, 0.01))),
                    "low": min(o, c) * (1 - abs(rng.normal(0, 0.01))),
                    "close": c,
                    "volume": max(100_000, int(rng.normal(5_000_000, 1_000_000))),
                }
            )

    df = pd.DataFrame(records)

    print("Calculating technical indicators...")
    df = add_indicators(df, TECHNICAL_INDICATORS)

    print("Synthesizing VIX and Sentiment...")
    market_returns = df.groupby("date")["close"].mean().pct_change().fillna(0)
    synthetic_vix = (20.0 - market_returns * 500 + rng.normal(0, 2, len(market_returns))).clip(10, 80)
    df["vix"] = df["date"].map(synthetic_vix.to_dict())

    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df["future_3d_close"] = df.groupby("tic")["close"].shift(-3)
    df["future_return"] = (df["future_3d_close"] - df["close"]) / df["close"]
    df["sentiment"] = np.where(df["future_return"] > 0.015, 0.8, np.where(df["future_return"] < -0.015, -0.8, 0.0))
    df = df.drop(columns=["future_3d_close", "future_return"])

    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(SYNTH_OUT), exist_ok=True)
    df.to_csv(SYNTH_OUT, index=False)
    print(f"Done. Saved to {SYNTH_OUT}")


if __name__ == "__main__":
    generate_market()
