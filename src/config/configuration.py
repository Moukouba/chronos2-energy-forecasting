"""
Configuration classes for energy forecasting pipeline
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import logging

from ..entity.config_entity import (
    DataConfigEntity,
    ModelConfigEntity,
    TrainingConfigEntity,
    PredictionConfigEntity,
    OutputConfigEntity,
    APIConfigEntity,
)

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str
    target_columns: List[str]
    lag_set: List[int]
    rolling_windows: List[int]
    train_split: float = 0.85
    freq: str = "H"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        return cls(
            data_path=config_dict.get("data_path"),
            target_columns=config_dict.get("target_columns", []),
            lag_set=config_dict.get("lag_set", [1, 2, 3, 6, 12, 24]),
            rolling_windows=config_dict.get("rolling_windows", [3, 6, 12, 24]),
            train_split=config_dict.get("train_split", 0.85),
            freq=config_dict.get("freq", "H")
        )
    
    def to_entity(self) -> DataConfigEntity:
        """Convert to entity object"""
        return DataConfigEntity(
            data_path=self.data_path,
            target_columns=self.target_columns,
            lag_set=self.lag_set,
            rolling_windows=self.rolling_windows,
            train_split=self.train_split,
            freq=self.freq
        )


@dataclass
class ModelConfig:
    """Model configuration"""
    input_chunk_length: int
    output_chunk_length: int
    hub_model_name: str
    random_state: int = 42
    enable_finetuning: Optional[Dict[str, List[str]]] = None
    likelihood_quantiles: List[float] = None
    
    def __post_init__(self):
        if self.enable_finetuning is None:
            self.enable_finetuning = {
                "unfreeze": ["*patch_embedding*", "*output_projection*"]
            }
        if self.likelihood_quantiles is None:
            self.likelihood_quantiles = [0.1, 0.5, 0.9]
    
    def to_entity(self) -> ModelConfigEntity:
        """Convert to entity object"""
        return ModelConfigEntity(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hub_model_name=self.hub_model_name,
            random_state=self.random_state,
            enable_finetuning=self.enable_finetuning,
            likelihood_quantiles=self.likelihood_quantiles
        )


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int
    n_epochs: int
    warmup_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int = 5
    monitor_metric: str = "val_loss"
    save_checkpoints: bool = True
    model_name: str = "chronos"
    optimizer: str = "adamw"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            batch_size=config_dict.get("batch_size", 4),
            n_epochs=config_dict.get("n_epochs", 20),
            warmup_epochs=config_dict.get("warmup_epochs", 2),
            learning_rate=config_dict.get("learning_rate", 1e-5),
            weight_decay=config_dict.get("weight_decay", 0.01),
            early_stopping_patience=config_dict.get("early_stopping_patience", 5),
            monitor_metric=config_dict.get("monitor_metric", "val_loss"),
            save_checkpoints=config_dict.get("save_checkpoints", True),
            model_name=config_dict.get("model_name", "chronos"),
            optimizer=config_dict.get("optimizer", "adamw")
        )
    
    def to_entity(self) -> TrainingConfigEntity:
        """Convert to entity object"""
        return TrainingConfigEntity(
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            warmup_epochs=self.warmup_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            early_stopping_patience=self.early_stopping_patience,
            monitor_metric=self.monitor_metric,
            save_checkpoints=self.save_checkpoints,
            model_name=self.model_name,
            optimizer=self.optimizer
        )


@dataclass
class PredictionConfig:
    """Prediction configuration"""
    output_chunk_length: int = 24
    predict_likelihood_parameters: bool = True
    num_chunks: int = 6
    
    def to_entity(self) -> PredictionConfigEntity:
        """Convert to entity object"""
        return PredictionConfigEntity(
            output_chunk_length=self.output_chunk_length,
            predict_likelihood_parameters=self.predict_likelihood_parameters,
            num_chunks=self.num_chunks
        )


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    models_dir: str = "models"
    
    def to_entity(self) -> OutputConfigEntity:
        """Convert to entity object"""
        return OutputConfigEntity(
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            logs_dir=self.logs_dir,
            models_dir=self.models_dir
        )


class ConfigManager:
    """Configuration manager for the pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_config = self._load_data_config()
        self.model_config = self._load_model_config()
        self.training_config = self._load_training_config()
        self.prediction_config = self._load_prediction_config()
        self.output_config = self._load_output_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {self.config_path}, using defaults")
        return {}
    
    def _load_data_config(self) -> DataConfig:
        """Load data configuration"""
        data_dict = self.config.get("data", {})
        return DataConfig.from_dict(data_dict)
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration"""
        model_dict = self.config.get("model", {})
        return ModelConfig(**model_dict)
    
    def _load_training_config(self) -> TrainingConfig:
        """Load training configuration"""
        training_dict = self.config.get("training", {})
        return TrainingConfig.from_dict(training_dict)
    
    def _load_prediction_config(self) -> PredictionConfig:
        """Load prediction configuration"""
        prediction_dict = self.config.get("prediction", {})
        return PredictionConfig(**prediction_dict)
    
    def _load_output_config(self) -> OutputConfig:
        """Load output configuration"""
        output_dict = self.config.get("output", {})
        return OutputConfig(**output_dict)
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        data_dict = self.config.get("data", {})
        return DataConfig.from_dict(data_dict)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_dict = self.config.get("model", {})
        return ModelConfig(
            input_chunk_length=model_dict.get("input_chunk_length", 168),
            output_chunk_length=model_dict.get("output_chunk_length", 24),
            hub_model_name=model_dict.get("hub_model_name", "autogluon/chronos-2-small"),
            random_state=model_dict.get("random_state", 42),
            enable_finetuning=model_dict.get("enable_finetuning"),
            likelihood_quantiles=model_dict.get("likelihood_quantiles"),
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        training_dict = self.config.get("training", {})
        return TrainingConfig.from_dict(training_dict)
    
    def get_prediction_config(self) -> PredictionConfig:
        """Get prediction configuration."""
        pred_dict = self.config.get("prediction", {})
        return PredictionConfig(**pred_dict) if pred_dict else PredictionConfig()
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as dictionaries."""
        return self.config
