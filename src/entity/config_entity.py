"""
Entity classes for configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfigEntity:
    """Data configuration entity"""
    data_path: str
    target_columns: List[str]
    lag_set: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    train_split: float = 0.85
    freq: str = "H"
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not 0 < self.train_split < 1:
            raise ValueError(f"train_split must be between 0 and 1, got {self.train_split}")


@dataclass
class ModelConfigEntity:
    """Model configuration entity"""
    input_chunk_length: int
    output_chunk_length: int
    hub_model_name: str
    random_state: int = 42
    enable_finetuning: Optional[Dict[str, List[str]]] = None
    likelihood_quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if self.input_chunk_length <= 0 or self.output_chunk_length <= 0:
            raise ValueError("Chunk lengths must be positive integers")
        if self.enable_finetuning is None:
            self.enable_finetuning = {
                "unfreeze": ["*patch_embedding*", "*output_projection*"]
            }


@dataclass
class TrainingConfigEntity:
    """Training configuration entity"""
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
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("batch_size and n_epochs must be positive integers")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive, weight_decay non-negative")


@dataclass
class PredictionConfigEntity:
    """Prediction configuration entity"""
    output_chunk_length: int = 24
    predict_likelihood_parameters: bool = True
    num_chunks: int = 6
    
    def __post_init__(self):
        """Validate entity after initialization"""
        if self.output_chunk_length <= 0 or self.num_chunks <= 0:
            raise ValueError("output_chunk_length and num_chunks must be positive")


@dataclass
class OutputConfigEntity:
    """Output configuration entity"""
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    models_dir: str = "models"
    
    def __post_init__(self):
        """Create output directories if they don't exist"""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir, self.models_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfigEntity:
    """API configuration entity"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
