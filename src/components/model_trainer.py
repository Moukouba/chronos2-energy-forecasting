"""
Model trainer component
"""

import functools
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    MeanAbsoluteError,
    MeanSquaredError,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR

from darts import TimeSeries
from darts.models import Chronos2Model
from darts.utils.likelihood_models import QuantileRegression


class LossCurveCallback(pl.Callback):
    """Callback to track loss curves during training."""
    
    def __init__(self):
        self.train_losses, self.train_epochs = [], []
        self.val_losses, self.val_epochs = [], []
        self.val_mape, self.mape_epochs = [], []
    
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(float(loss))
            self.train_epochs.append(trainer.current_epoch)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(float(loss))
            self.val_epochs.append(trainer.current_epoch)
        
        mape = trainer.callback_metrics.get("val_MAPE")
        if mape is not None:
            self.val_mape.append(float(mape))
            self.mape_epochs.append(trainer.current_epoch)


class ModelTrainer:
    """Train Chronos2 model for time series forecasting"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        self.model: Optional[Chronos2Model] = None
        self.loss_cb: Optional[LossCurveCallback] = None
    
    def _warmup_cosine(self, epoch: int, warmup_epochs: int, total_epochs: int) -> float:
        """Warmup-cosine learning rate scheduler."""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def _create_early_stopping(self) -> EarlyStopping:
        """Create early stopping callback."""
        return EarlyStopping(
            monitor=self.training_config.get("monitor_metric", "val_loss"),
            patience=self.training_config.get("early_stopping_patience", 5),
            mode="min",
        )
    
    def _create_torch_metrics(self) -> MetricCollection:
        """Create torch metrics collection."""
        return MetricCollection({
            "MAPE": MeanAbsolutePercentageError(),
            "MAE": MeanAbsoluteError(),
            "RMSE": MeanSquaredError(squared=False),
        })
    
    def _create_loss_callback(self) -> LossCurveCallback:
        """Create loss curve callback."""
        return LossCurveCallback()
    
    def _create_add_encoders(self) -> Dict[str, Any]:
        """Create add_encoders configuration."""
        def encode_is_weekend(idx):
            return (idx.dayofweek >= 5).astype(float)
        
        def encode_morning_peak(idx):
            return ((idx.hour >= 6) & (idx.hour <= 10)).astype(float)
        
        def encode_evening_peak(idx):
            return ((idx.hour >= 17) & (idx.hour <= 21)).astype(float)
        
        def daily_sin_2(idx):
            return np.sin(2 * np.pi * 2 * idx.hour / 24)
        
        def daily_cos_2(idx):
            return np.cos(2 * np.pi * 2 * idx.hour / 24)
        
        def daily_sin_3(idx):
            return np.sin(2 * np.pi * 3 * idx.hour / 24)
        
        def daily_cos_3(idx):
            return np.cos(2 * np.pi * 3 * idx.hour / 24)
        
        def weekly_sin_2(idx):
            return np.sin(2 * np.pi * 2 * (idx.dayofweek * 24 + idx.hour) / (24 * 7))
        
        def weekly_cos_2(idx):
            return np.cos(2 * np.pi * 2 * (idx.dayofweek * 24 + idx.hour) / (24 * 7))
        
        from darts.dataprocessing.transformers import Scaler
        
        return {
            "cyclic": {"future": ["hour", "dayofweek", "month"]},
            "custom": {
                "future": [
                    encode_is_weekend,
                    encode_morning_peak, encode_evening_peak,
                    daily_sin_2, daily_cos_2,
                    daily_sin_3, daily_cos_3,
                    weekly_sin_2, weekly_cos_2,
                ]
            },
            "transformer": Scaler(),
        }
    
    def build_model(self) -> Chronos2Model:
        """Build Chronos2 model."""
        self.loss_cb = self._create_loss_callback()
        
        model = Chronos2Model(
            input_chunk_length=self.model_config.get("input_chunk_length", 168),
            output_chunk_length=self.model_config.get("output_chunk_length", 24),
            random_state=self.model_config.get("random_state", 42),
            hub_model_name=self.model_config.get("hub_model_name", "autogluon/chronos-2-small"),
            
            enable_finetuning=self.model_config.get(
                "enable_finetuning", 
                {"unfreeze": ["*patch_embedding*", "*output_projection*"]}
            ),
            
            add_encoders=self._create_add_encoders(),
            
            likelihood=QuantileRegression(
                quantiles=self.model_config.get("likelihood_quantiles", [0.1, 0.5, 0.9])
            ),
            
            save_checkpoints=self.training_config.get("save_checkpoints", True),
            model_name=self.training_config.get("model_name", "chronos"),
            force_reset=True,
            
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": self.training_config.get("learning_rate", 1e-5),
                "weight_decay": self.training_config.get("weight_decay", 0.01),
            },
            
            lr_scheduler_cls=LambdaLR,
            lr_scheduler_kwargs={
                "lr_lambda": functools.partial(
                    self._warmup_cosine,
                    warmup_epochs=self.training_config.get("warmup_epochs", 2),
                    total_epochs=self.training_config.get("n_epochs", 20),
                )
            },
            
            nr_epochs_val_period=1,
            batch_size=self.training_config.get("batch_size", 4),
            
            pl_trainer_kwargs={
                "accelerator": "auto",
                "devices": "auto",
                "strategy": "auto",
                "gradient_clip_val": 0.5,
                "gradient_clip_algorithm": "norm",
                "callbacks": [self._create_early_stopping(), self.loss_cb],
                "enable_progress_bar": True,
                "precision": "32",
            },
        )
        
        self.model = model
        return model
    
    def train(self, model: Chronos2Model, 
             train_series: TimeSeries,
             train_past_cov: TimeSeries,
             train_future_cov: TimeSeries,
             val_series: TimeSeries = None,
             val_past_cov: TimeSeries = None,
             val_future_cov: TimeSeries = None,
             epochs: int = 20,
             load_best: bool = True) -> Chronos2Model:
        """Train the model."""
        model.fit(
            series=train_series,
            past_covariates=train_past_cov,
            future_covariates=train_future_cov,
            val_series=val_series,
            val_past_covariates=val_past_cov,
            val_future_covariates=val_future_cov,
            epochs=epochs,
            load_best=load_best,
            verbose=True,
        )
        return model
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        """Get loss history from training."""
        if self.loss_cb is None:
            return {}
        
        return {
            "train_losses": self.loss_cb.train_losses,
            "train_epochs": self.loss_cb.train_epochs,
            "val_losses": self.loss_cb.val_losses,
            "val_epochs": self.loss_cb.val_epochs,
            "val_mape": self.loss_cb.val_mape,
            "mape_epochs": self.loss_cb.mape_epochs,
        }
    
    def save_model(self, model: Chronos2Model, filepath: str) -> None:
        """Save model to file."""
        model.save(filepath)
    
    def save_onnx(self, model: Chronos2Model, filepath: str) -> None:
        """Save model in ONNX format."""
        model.to_onnx(filepath, export_params=True)
