import shap
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class XAIExplainer:
    """
    Task 19: Explainable AI (XAI)
    Uses SHAP (SHapley Additive exPlanations) to crack open the neural network "black box" 
    and output a human-readable explanation of WHY the AI made a specific trade.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def generate_explanation(self, model_predict_fn, background_data, current_state):
        """
        model_predict_fn: A wrapper function that takes a numpy array and returns action logits/probabilities.
        background_data: A sample of historical states (e.g. 100 rows) to give SHAP a baseline.
        current_state: The exact state vector the AI just looked at before making its trade.
        """
        print("\n[XAI] Analyzing neural network activation pathways via SHAP...")
        
        # We use KernelExplainer since we are dealing with complex Custom PyTorch models 
        # (CNNs, Transformers, etc) that might not work with DeepExplainer out of the box.
        explainer = shap.KernelExplainer(model_predict_fn, background_data)
        
        # Calculate SHAP values for the current state
        shap_values = explainer.shap_values(current_state, nsamples=100)
        
        # shap_values is a list for multi-action outputs. Let's assume we care about the magnitude
        # We aggregate the absolute SHAP values across all output actions to see which input feature
        # had the biggest impact on the overall decision matrix.
        if isinstance(shap_values, list):
            # Sum absolute impact across all stocks the bot is trying to trade
            aggregated_impact = np.sum(np.abs(shap_values), axis=0).flatten()
        else:
            aggregated_impact = np.abs(shap_values).flatten()
            
        # Map impacts to feature names
        impact_map = dict(zip(self.feature_names, aggregated_impact))
        
        # Sort features by highest impact
        sorted_impact = sorted(impact_map.items(), key=lambda item: item[1], reverse=True)
        
        print("[XAI] --- TRADE EXPLANATION SUMMARY ---")
        total_impact = sum(aggregated_impact)
        if total_impact == 0: total_impact = 1e-8 # prevent div/0
        
        # Print the top 3 reasons for the trade
        for i in range(min(3, len(sorted_impact))):
            feature, impact = sorted_impact[i]
            percentage = (impact / total_impact) * 100
            print(f"[XAI] Reason {i+1}: {feature} ({percentage:.1f}% confidence impact)")
            
        return sorted_impact

if __name__ == "__main__":
    # Dummy Test
    dummy_names = ["MACD", "RSI_30", "VIX", "Sentiment", "Close_SMA_30"]
    explainer = XAIExplainer(dummy_names)
    print("XAI Explainer initialized successfully.")
