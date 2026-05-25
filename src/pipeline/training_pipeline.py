"""
Training pipeline
"""

from typing import Dict, Any, Optional
from pathlib import Path

from darts import TimeSeries

from ..components.data_processor import DataProcessor
from ..components.model_trainer import ModelTrainer
from ..components.predictor import Predictor
from ..components.evaluator import Evaluator


class TrainingPipeline:
    """Complete training pipeline for energy forecasting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_processor: Optional[DataProcessor] = None
        self.model_trainer: Optional[ModelTrainer] = None
        self.predictor: Optional[Predictor] = None
        self.evaluator: Optional[Evaluator] = None
        self.model: Optional[TimeSeries] = None
        
    def setup(self) -> None:
        """Setup pipeline components."""
        data_config = self.config.get("data", {})
        model_config = self.config.get("model", {})
        training_config = self.config.get("training", {})
        
        self.data_processor = DataProcessor(
            target_columns=data_config.get("target_columns", []),
            lag_set=data_config.get("lag_set", [1, 2, 3, 6, 12, 24]),
            rolling_windows=data_config.get("rolling_windows", [3, 6, 12, 24]),
        )
        
        self.model_trainer = ModelTrainer(
            model_config=model_config,
            training_config=training_config,
        )
        
        self.predictor = Predictor(
            output_chunk_length=model_config.get("output_chunk_length", 24)
        )
        
        self.evaluator = Evaluator()
    
    def run(self, data_path: str, output_dir: str = "outputs") -> Dict[str, Any]:
        """Run complete training pipeline."""
        import os
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("STARTING TRAINING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load and process data
        print("\n[1/5] Loading and processing data...")
        df = self.data_processor.load_data(data_path)
        print(f"  Loaded {len(df)} rows")
        
        print("  Building features...")
        df = self.data_processor.build_features(df)
        print(f"  Total features: {len(df.columns)}")
        
        print("  Classifying covariates...")
        past_cov, future_cov = self.data_processor.classify_covariates(df)
        print(f"  Past covariates: {len(past_cov)}")
        print(f"  Future covariates: {len(future_cov)}")
        
        # Step 2: Split data
        print("\n[2/5] Splitting data...")
        split_idx = int(self.config.get("data", {}).get("train_split", 0.85) * len(df))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        
        # Step 3: Create TimeSeries objects
        print("\n[3/5] Creating TimeSeries objects...")
        train_series, train_past_cov, train_future_cov = \
            self.data_processor.create_time_series(train_df)
        val_series, val_past_cov, val_future_cov = \
            self.data_processor.create_time_series(val_df)
        
        # Step 4: Scale covariates
        print("\n[4/5] Scaling covariates...")
        train_past_scaled, train_future_scaled, val_past_scaled, val_future_scaled = \
            self.data_processor.scale_covariates(
                train_past_cov, train_future_cov, val_past_cov, val_future_cov
            )
        
        # Step 5: Build and train model
        print("\n[5/5] Building and training model...")
        self.model_trainer.build_model()
        
        self.model = self.model_trainer.train(
            self.model_trainer.model,
            train_series, train_past_scaled, train_future_scaled,
            val_series, val_past_scaled, val_future_scaled,
            epochs=self.config.get("training", {}).get("n_epochs", 20),
            load_best=True,
        )
        
        # Save model
        model_path = output_dir / "model.pt"
        onnx_path = output_dir / "model.onnx"
        
        print(f"\n  Saving model to {model_path}...")
        self.model_trainer.save_model(self.model, str(model_path))
        
        print(f"  Saving ONNX model to {onnx_path}...")
        self.model_trainer.save_onnx(self.model, str(onnx_path))
        
        # Evaluate
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)
        
        target_columns = self.config.get("data", {}).get("target_columns", [])
        hub_results = {}
        
        for target_col in target_columns:
            print(f"\n  Evaluating {target_col}...")
            pred_multi, val_actuals = self.predictor.rolling_rollout_predict(
                self.model, train_series, val_series,
                train_past_scaled, val_past_scaled,
                train_future_scaled, val_future_scaled,
                target_col=target_col,
                num_chunks=6,
                seed=421,
            )
            
            results = self.evaluator.evaluate_hub(
                pred_multi, val_actuals, target_col,
                output_chunk_length=self.config.get("model", {}).get("output_chunk_length", 24)
            )
            hub_results[target_col] = results
            
            print(f"    Overall MAPE: {results['overall_mape']:.2f}%")
        
        # Save evaluation results
        summary_df = self.evaluator.create_summary_table(hub_results)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string())
        
        results_path = output_dir / "evaluation_results.json"
        self.evaluator.save_evaluation_results(hub_results, str(results_path))
        
        print(f"\n  Evaluation results saved to {results_path}")
        
        return {
            "model_path": str(model_path),
            "onnx_path": str(onnx_path),
            "evaluation_results": hub_results,
            "summary": summary_df.to_dict(),
        }
