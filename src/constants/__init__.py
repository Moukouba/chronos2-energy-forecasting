"""
Constants for the energy forecasting pipeline
"""

# Project info
PROJECT_NAME = "energy_forecast"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Production-ready energy forecasting pipeline using Chronos2"

# Default paths
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_DATA_PATH = "/home/moukouba/equilibrium/model_ready.parquet"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_MODELS_DIR = "models"

# Model constants
DEFAULT_MODEL_NAME = "chronos"
DEFAULT_INPUT_CHUNK_LENGTH = 168
DEFAULT_OUTPUT_CHUNK_LENGTH = 24
DEFAULT_HUB_MODEL_NAME = "autogluon/chronos-2-small"

# Training constants
DEFAULT_BATCH_SIZE = 4
DEFAULT_N_EPOCHS = 20
DEFAULT_WARMUP_EPOCHS = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_OPTIMIZER = "adamw"

# Prediction constants
DEFAULT_NUM_CHUNKS = 6
DEFAULT_PREDICT_LIKELIHOOD = True
DEFAULT_LIKELIHOOD_QUANTILES = [0.1, 0.5, 0.9]

# API constants
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Data constants
DEFAULT_TRAIN_SPLIT = 0.85
DEFAULT_FREQ = "H"  # Hourly frequency
DEFAULT_LAG_SET = [1, 2, 3, 6, 12, 24]
DEFAULT_ROLLING_WINDOWS = [3, 6, 12, 24]

# Target columns
DEFAULT_TARGET_COLUMNS = [
    "da_energy_aeci_lmpexpost_ac",
    "da_energy_michigan_hub_lmpexpost_ac",
    "da_energy_minn_hub_lmpexpost_ac",
]

# Evaluation metrics
MAPE_THRESHOLD = 15.0  # 15% MAPE threshold
    "da_energy_minn_hub_lmpexpost_ac",
]
DEFAULT_MODEL_NAME = "autogluon/chronos-2-small"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_LOGS_DIR = "logs"

__all__ = [
    "PROJECT_NAME",
    "VERSION",
    "DEFAULT_DATA_PATH",
    "DEFAULT_TARGET_COLUMNS",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_LOGS_DIR",
]
