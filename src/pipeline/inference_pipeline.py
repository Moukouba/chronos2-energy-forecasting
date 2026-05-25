"""
Inference pipeline
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from darts import TimeSeries

from ..components.data_processor import DataProcessor
from ..components.predictor import Predictor
from ..components.evaluator import Evaluator


class InferencePipeline:
    """Inference pipeline for making predictions with trained model"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.data_processor: Optional[DataProcessor] = None
        self.predictor: Optional[Predictor] = None
        self.evaluator: Optional[Evaluator] = None
        
    def load_model(self) -> None:
        """Load trained model."""
        from darts.models import Chronos2Model
        
        print(f"Loading model from {self.model_path}...")
        self.model = Chronos2Model.load(self.model_path)
        print("Model loaded successfully!")
        
        # Setup components
        data_config = self.config.get("data", {})
        self.data_processor = DataProcessor(
            target_columns=data_config.get("target_columns", []),
            lag_set=data_config.get("lag_set", [1, 2, 3, 6, 12, 24]),
            rolling_windows=data_config.get("rolling_windows", [3, 6, 12, 24]),
        )
        
        self.predictor = Predictor(
            output_chunk_length=self.config.get("model", {}).get("output_chunk_length", 24)
        )
        
        self.evaluator = Evaluator()
    
    def predict(self, data_path: str, target_col: str,
               num_chunks: int = 6, seed: Optional[int] = None) -> Dict[str, Any]:
        """Make predictions on new data."""
        import random
        
        # Load data
        df = self.data_processor.load_data(data_path)
        df = self.data_processor.build_features(df)
        past_cov, future_cov = self.data_processor.classify_covariates(df)
        
        # Create TimeSeries
        full_series, full_past_cov, full_future_cov = \
            self.data_processor.create_time_series(df)
        
        # Split
        split_idx = int(self.config.get("data", {}).get("train_split", 0.85) * len(df))
        train_series = full_series[:split_idx]
        val_series = full_series[split_idx:]
        train_past_cov = full_past_cov[:split_idx]
        val_past_cov = full_past_cov[split_idx:]
        train_future_cov = full_future_cov[:split_idx]
        val_future_cov = full_future_cov[split_idx:]
        
        # Scale
        train_past_scaled, train_future_scaled, val_past_scaled, val_future_scaled = \
            self.data_processor.scale_covariates(
                train_past_cov, train_future_cov, val_past_cov, val_future_cov
            )
        
        # Predict
        pred_multi, val_actuals = self.predictor.rolling_rollout_predict(
            self.model, train_series, val_series,
            train_past_scaled, val_past_scaled,
            train_future_scaled, val_future_scaled,
            target_col=target_col,
            num_chunks=num_chunks,
            seed=seed,
        )
        
        # Evaluate
        results = self.evaluator.evaluate_hub(
            pred_multi, val_actuals, target_col,
            output_chunk_length=self.config.get("model", {}).get("output_chunk_length", 24)
        )
        
        return {
            "predictions": pred_multi,
            "actuals": val_actuals,
            "metrics": results,
        }
    
    def batch_predict(self, data_paths: List[str], target_cols: List[str],
                     num_chunks: int = 6) -> Dict[str, Dict[str, Any]]:
        """Make batch predictions on multiple files."""
        results = {}
        
        for data_path in data_paths:
            print(f"\nProcessing {data_path}...")
            for target_col in target_cols:
                key = f"{Path(data_path).stem}_{target_col}"
                results[key] = self.predict(
                    data_path=data_path,
                    target_col=target_col,
                    num_chunks=num_chunks,
                )
        
        return results
