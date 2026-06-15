import os
from datetime import datetime

import torch
import pandas as pd
import yfinance as yf
from transformers import pipeline

from algo_v2.config import TICKER_LIST

DB_PATH = 'sentiment_db.csv'
TODAY = datetime.today().strftime('%Y-%m-%d')

print("Loading FinBERT model...")
# Using the standard financial sentiment model
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Error loading FinBERT: {e}")
    print("Ensure you have installed transformers and torch: pip install transformers torch")
    exit(1)

def get_sentiment_score(texts):
    if not texts:
        return 0.0
    results = sentiment_pipeline(texts)
    score = 0.0
    for res in results:
        label = res['label'] # positive, negative, neutral
        prob = res['score']
        if label == 'positive':
            score += prob
        elif label == 'negative':
            score -= prob
    return score / len(results)

print(f"Scraping news for {TODAY}...")
new_records = []

for tic in TICKER_LIST:
    try:
        stock = yf.Ticker(tic)
        news = stock.news
        headlines = [item['title'] for item in news if 'title' in item]
        
        if len(headlines) == 0:
            sentiment = 0.0
        else:
            sentiment = get_sentiment_score(headlines)
            
        new_records.append({
            'date': TODAY,
            'tic': tic,
            'sentiment': sentiment
        })
        print(f"{tic}: {sentiment:.2f} (from {len(headlines)} articles)")
    except Exception as e:
        print(f"Error processing {tic}: {e}")
        new_records.append({
            'date': TODAY,
            'tic': tic,
            'sentiment': 0.0
        })

new_df = pd.DataFrame(new_records)

if os.path.exists(DB_PATH):
    old_df = pd.read_csv(DB_PATH)
    # Remove existing entries for today to prevent duplicates
    old_df = old_df[old_df['date'] != TODAY]
    final_df = pd.concat([old_df, new_df], ignore_index=True)
else:
    final_df = new_df

final_df.to_csv(DB_PATH, index=False)
print(f"Sentiment DB updated successfully at {DB_PATH}.")
