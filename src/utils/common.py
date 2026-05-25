"""
Utility functions and common helpers
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


# Configure logging
def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"energy_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, filepath: str) -> None:
    """Save data as JSON file."""
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """Save data as pickle file."""
    ensure_dir(str(Path(filepath).parent))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle file."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_file_list(directory: str, extension: str = ".parquet") -> List[str]:
    """Get list of files with specific extension."""
    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def calculate_mape(actual: np.ndarray, pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value as percentage
    """
    actual, pred = np.asarray(actual), np.asarray(pred)
    denom = np.where(np.abs(actual) < epsilon, epsilon, np.abs(actual))
    return float(np.mean(np.abs(actual - pred) / denom) * 100)


def calculate_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.mean(np.abs(actual - pred)))


def calculate_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def calculate_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculate multiple error metrics.
    
    Returns:
        Dictionary containing MAPE, MAE, and RMSE
    """
    return {
        "mape": calculate_mape(actual, pred),
        "mae": calculate_mae(actual, pred),
        "rmse": calculate_rmse(actual, pred),
    }


def format_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format metrics as DataFrame for display."""
    if not metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(metrics).T


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def load_data(filepath: str, engine: str = "fastparquet") -> pd.DataFrame:
    """Load data from parquet file.
    
    Args:
        filepath: Path to parquet file
        engine: Engine to use (fastparquet or pyarrow)
        
    Returns:
        Loaded DataFrame
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    return pd.read_parquet(filepath, engine=engine)


def save_data(df: pd.DataFrame, filepath: str, engine: str = "fastparquet") -> None:
    """Save DataFrame to parquet file."""
    ensure_dir(str(Path(filepath).parent))
    df.to_parquet(filepath, engine=engine, index=False)


def print_section(title: str, width: int = 80) -> None:
    """Print formatted section header."""
    print("=" * width)
    print(f"{title.center(width)}")
    print("=" * width)
