"""
Energy Forecasting Pipeline Package
Production-ready time series forecasting system using Chronos2
"""

__version__ = "1.0.0"
__author__ = "Energy Forecasting Team"
__description__ = "Production-ready time series energy forecasting using Chronos2"

from .config.configuration import ConfigManager
from .pipeline.training_pipeline import TrainingPipeline
from .pipeline.inference_pipeline import InferencePipeline
from .utils.common import setup_logging

__all__ = [
    "ConfigManager",
    "TrainingPipeline",
    "InferencePipeline",
    "setup_logging",
]
