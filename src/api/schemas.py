"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for prediction"""
    data_path: str = Field(..., description="Path to data file")
    target_column: str = Field(..., description="Target column name")
    num_chunks: int = Field(default=6, description="Number of prediction chunks")
    seed: Optional[int] = Field(default=None, description="Random seed")
    
    class Config:
        schema_extra = {
            "example": {
                "data_path": "/path/to/data.parquet",
                "target_column": "da_energy_aeci_lmpexpost_ac",
                "num_chunks": 6,
                "seed": 42
            }
        }


class TrainingRequest(BaseModel):
    """Request schema for training"""
    data_path: str = Field(..., description="Path to training data")
    output_dir: str = Field(default="outputs", description="Output directory")
    config_path: str = Field(default="config/config.yaml", description="Config file path")
    
    class Config:
        schema_extra = {
            "example": {
                "data_path": "/path/to/data.parquet",
                "output_dir": "outputs",
                "config_path": "config/config.yaml"
            }
        }


class MetricResponse(BaseModel):
    """Response schema for metrics"""
    overall_mape: float = Field(..., description="Overall Mean Absolute Percentage Error")
    chunk_mape: List[float] = Field(..., description="Per-chunk MAPE values")
    mae: Optional[float] = Field(default=None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(default=None, description="Root Mean Squared Error")


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    message: str = Field(..., description="Success message")
    target_column: str = Field(..., description="Target column name")
    num_predictions: int = Field(..., description="Number of predictions")
    metrics: MetricResponse
    predictions_shape: List[int] = Field(..., description="Shape of predictions array")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingResponse(BaseModel):
    """Response schema for training"""
    message: str = Field(..., description="Success message")
    model_path: str = Field(..., description="Path to saved model")
    metrics: Dict[str, Any] = Field(..., description="Training metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0")


class ConfigResponse(BaseModel):
    """Response schema for configuration info"""
    data_path: str
    target_columns: List[str]
    model_name: str
    batch_size: int
    n_epochs: int
    learning_rate: float


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
