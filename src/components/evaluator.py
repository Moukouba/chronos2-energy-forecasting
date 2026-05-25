"""
Evaluator component
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from darts import TimeSeries


class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def calculate_mape(self, actual: np.ndarray, pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        actual, pred = np.asarray(actual), np.asarray(pred)
        denom = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
        return float(np.mean(np.abs(actual - pred) / denom) * 100)
    
    def evaluate_hub(self, pred_multi: TimeSeries, val_actuals: TimeSeries,
                    target_col: str, output_chunk_length: int = 24) -> Dict[str, Any]:
        """Evaluate performance for a single hub."""
        pred_df = pred_multi.pd_dataframe()
        actual_df = val_actuals.pd_dataframe()
        
        q50_col = [c for c in pred_df.columns if target_col in c and "q0.500" in c][0]
        
        actual_values = actual_df[target_col].values
        pred_values = pred_df[q50_col].values
        
        overall_mape = self.calculate_mape(actual_values, pred_values)
        
        chunk_mape = []
        for k in range(6):
            start = k * output_chunk_length
            end = (k + 1) * output_chunk_length
            chunk_mape.append(self.calculate_mape(
                actual_values[start:end],
                pred_values[start:end]
            ))
        
        return {
            "target_col": target_col,
            "overall_mape": overall_mape,
            "chunk_mape": chunk_mape,
            "q50_column": q50_col,
        }
    
    def create_summary_table(self, hub_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create summary table of all hub results."""
        summary_rows = []
        
        for hub, results in hub_results.items():
            chunk_mapes = results.get("chunk_mape", [])
            overall = results.get("overall_mape", 0)
            
            row = {
                "Hub": hub.replace("da_energy_", "").replace("_lmpexpost_ac", ""),
                **{f"Chunk {k+1}": f"{v:.1f}%" for k, v in enumerate(chunk_mapes)},
                "Overall": f"{overall:.1f}%",
            }
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows).set_index("Hub")
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to file."""
        import json
        
        formatted_results = {}
        for hub, data in results.items():
            formatted_results[hub] = {
                "overall_mape": data.get("overall_mape"),
                "chunk_mape": data.get("chunk_mape"),
                "q50_column": data.get("q50_column"),
            }
        
        with open(filepath, 'w') as f:
            json.dump(formatted_results, f, indent=2)
