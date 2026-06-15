import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.exceptions import OptimizationError

class SafetyOptimizer:
    """
    Task 18: Mean-Variance Portfolio Optimization
    Acts as a mathematical safety net over the entire AI system.
    If the AI requests a portfolio that exceeds the 15% annual volatility limit,
    this script forcefully rebalances the allocation using PyPortfolioOpt to obey the limit.
    """
    def __init__(self, max_volatility=0.15):
        self.max_volatility = max_volatility

    def enforce_safety_limit(self, ai_requested_weights, historical_prices_df):
        """
        ai_requested_weights: dict of {ticker: weight} the 35 bots voted for.
        historical_prices_df: DataFrame of recent stock closing prices.
        """
        # Filter prices to only the requested stocks
        tickers = list(ai_requested_weights.keys())
        prices = historical_prices_df[tickers]
        
        # Calculate Expected Returns and Sample Covariance
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        # Check current volatility of the AI's requested portfolio
        weights_array = np.array([ai_requested_weights[t] for t in tickers])
        # Annualized volatility = sqrt( w.T * S * w )
        current_variance = np.dot(weights_array.T, np.dot(S, weights_array))
        current_volatility = np.sqrt(current_variance)
        
        print(f"[OPTIMIZER] AI Requested Volatility: {current_volatility*100:.2f}% (Limit: {self.max_volatility*100}%)")
        
        if current_volatility <= self.max_volatility:
            # Trade is safe, approve it
            print("[OPTIMIZER] Trade APPROVED. Volatility is within safe limits.")
            return ai_requested_weights
            
        print("[OPTIMIZER] VETO! Volatility limit exceeded. Rebalancing via PyPortfolioOpt...")
        
        # Build Efficient Frontier to find the maximum return portfolio FOR the max_volatility
        ef = EfficientFrontier(mu, S)
        
        try:
            # We add a constraint to force weights to be non-negative (long only)
            # and sum to 1 (fully invested)
            # efficiently finding the portfolio with volatility <= target
            raw_weights = ef.efficient_risk(target_volatility=self.max_volatility)
            cleaned_weights = ef.clean_weights()
            print("[OPTIMIZER] Rebalancing SUCCESSFUL. Trade modified for safety.")
            return cleaned_weights
            
        except OptimizationError:
            # If PyPortfolioOpt physically cannot find a portfolio with < 15% volatility 
            # (e.g. during a catastrophic market crash where everything is volatile),
            # we dump everything into cash.
            print("[OPTIMIZER] FAILED TO REBALANCE. The entire market is too volatile. Liquidating to CASH.")
            return {t: 0.0 for t in tickers} # 100% Cash position

if __name__ == "__main__":
    # Quick dummy test
    optimizer = SafetyOptimizer(max_volatility=0.15)
    print("Safety Optimizer initialized successfully.")
