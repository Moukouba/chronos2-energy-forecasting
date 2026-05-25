"""
Predictor component
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate


class Predictor:
    """Make predictions using trained Chronos2 model"""
    
    def __init__(self, output_chunk_length: int = 24):
        self.output_chunk_length = output_chunk_length
    
    def rolling_rollout_predict(self, model, train_series: TimeSeries,
                               val_series: TimeSeries,
                               train_past_cov: TimeSeries,
                               val_past_cov: TimeSeries,
                               train_future_cov: TimeSeries,
                               val_future_cov: TimeSeries,
                               target_col: str,
                               num_chunks: int = 6,
                               seed: Optional[int] = None) -> Tuple[TimeSeries, TimeSeries]:
        """Perform rolling rollout prediction."""
        import random
        
        if seed is not None:
            random.seed(seed)
        
        def _safe_concat(a, b):
            try:
                return a.concatenate(b, ignore_time_axis=False)
            except:
                return a.concatenate(b, ignore_time_axis=True)
        
        full_series = _safe_concat(train_series, val_series)
        full_past_cov = _safe_concat(train_past_cov, val_past_cov)
        full_future_cov = _safe_concat(train_future_cov, val_future_cov)
        
        L_train = len(train_series)
        L_val = len(val_series)
        total_needed = num_chunks * self.output_chunk_length
        
        if L_val < total_needed:
            raise ValueError(
                f"Validation set too short ({L_val}h) for {num_chunks} chunks "
                f"of {self.output_chunk_length}h ({total_needed}h needed)."
            )
        
        start_idx = random.randint(0, L_val - total_needed)
        print(f"▶ Random start index in val_series : {start_idx}")
        print(f"▶ Generating {num_chunks} × {self.output_chunk_length}h predictions…")
        
        preds_list, actuals_list = [], []
        
        for k in range(num_chunks):
            history_end = L_train + start_idx + k * self.output_chunk_length
            history = full_series[:history_end]
            
            pred = model.predict(
                n=self.output_chunk_length,
                series=history,
                past_covariates=full_past_cov,
                future_covariates=full_future_cov,
                predict_likelihood_parameters=True,
            )
            preds_list.append(pred)
            
            actual_chunk = val_series[
                start_idx + k * self.output_chunk_length:
                start_idx + (k + 1) * self.output_chunk_length
            ]
            actuals_list.append(actual_chunk)
        
        pred_multi = concatenate(preds_list, axis=0)
        val_actuals = concatenate(actuals_list, axis=0)
        
        return pred_multi, val_actuals
    
    def predict_single(self, model, series: TimeSeries, 
                      past_covariates: TimeSeries = None,
                      future_covariates: TimeSeries = None,
                      n: int = 24) -> TimeSeries:
        """Make single prediction."""
        return model.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            predict_likelihood_parameters=True,
        )
    
    def calculate_mape(self, actual: np.ndarray, pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        actual, pred = np.asarray(actual), np.asarray(pred)
        denom = np.where(np.abs(actual) < 1e-8, 1e-8, np.abs(actual))
        return float(np.mean(np.abs(actual - pred) / denom) * 100)
    
    def evaluate_predictions(self, pred_multi: TimeSeries, 
                            val_actuals: TimeSeries,
                            target_col: str) -> Dict[str, float]:
        """Evaluate predictions and return metrics."""
        pred_df = pred_multi.pd_dataframe()
        actual_df = val_actuals.pd_dataframe()
        
        q50_col = [c for c in pred_df.columns if target_col in c and "q0.500" in c][0]
        
        actual_values = actual_df[target_col].values
        pred_values = pred_df[q50_col].values
        
        overall_mape = self.calculate_mape(actual_values, pred_values)
        
        # Chunk-wise MAPE
        chunk_mape = []
        for k in range(6):
            start = k * self.output_chunk_length
            end = (k + 1) * self.output_chunk_length
            chunk_mape.append(self.calculate_mape(
                actual_values[start:end],
                pred_values[start:end]
            ))
        
        return {
            "overall_mape": overall_mape,
            "chunk_mape": chunk_mape,
            "q50_column": q50_col,
        }
