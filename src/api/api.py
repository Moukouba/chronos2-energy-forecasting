"""
FastAPI application for energy forecasting
"""

import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from ..config.configuration import ConfigManager
from ..pipeline.training_pipeline import TrainingPipeline
from ..pipeline.inference_pipeline import InferencePipeline
from ..utils.common import setup_logging

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
    HealthResponse,
    ConfigResponse,
    ErrorResponse,
    MetricResponse,
)

# Setup logging
logger = setup_logging()

# Global state
app_state = {
    "config_manager": None,
    "inference_pipeline": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    try:
        logger.info("Initializing Energy Forecasting API")
        config_manager = ConfigManager("config/config.yaml")
        app_state["config_manager"] = config_manager
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Energy Forecasting API")


# Create FastAPI app
app = FastAPI(
    title="Energy Forecasting API",
    description="REST API for time series energy forecasting using Chronos2",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid value", "detail": str(exc)},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle FileNotFoundError exceptions"""
    logger.error(f"FileNotFoundError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "File not found", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# Configuration endpoints
@app.get(
    "/config",
    response_model=ConfigResponse,
    tags=["Configuration"],
    summary="Get current configuration"
)
async def get_config():
    """Get current configuration"""
    if not app_state["config_manager"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Configuration not loaded"
        )
    
    config = app_state["config_manager"]
    return {
        "data_path": config.data_config.data_path,
        "target_columns": config.data_config.target_columns,
        "model_name": config.model_config.hub_model_name,
        "batch_size": config.training_config.batch_size,
        "n_epochs": config.training_config.n_epochs,
        "learning_rate": config.training_config.learning_rate,
    }


# Training endpoint
@app.post(
    "/train",
    response_model=TrainingResponse,
    tags=["Training"],
    summary="Train the forecasting model",
    status_code=status.HTTP_202_ACCEPTED
)
async def train_model(request: TrainingRequest):
    """Train the model with provided configuration"""
    try:
        logger.info(f"Starting training with data: {request.data_path}")
        
        if not Path(request.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {request.data_path}")
        
        # Load config
        config_manager = ConfigManager(request.config_path)
        config_dict = config_manager.config
        
        # Create and run training pipeline
        training_pipeline = TrainingPipeline(config_dict)
        training_pipeline.setup()
        
        results = training_pipeline.run(
            data_path=request.data_path,
            output_dir=request.output_dir
        )
        
        logger.info("Training completed successfully")
        
        return {
            "message": "Training completed successfully",
            "model_path": results.get("model_path", ""),
            "metrics": results.get("metrics", {}),
        }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Make predictions"
)
async def make_prediction(request: PredictionRequest):
    """Make predictions with trained model"""
    try:
        logger.info(f"Processing prediction request for {request.target_column}")
        
        if not app_state["config_manager"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configuration not loaded"
            )
        
        if not Path(request.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {request.data_path}")
        
        config = app_state["config_manager"].config
        model_path = config.get("output", {}).get("checkpoint_dir", "checkpoints") + "/best_model"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        # Create inference pipeline
        inference_pipeline = InferencePipeline(model_path, config)
        inference_pipeline.load_model()
        
        # Make predictions
        results = inference_pipeline.predict(
            data_path=request.data_path,
            target_col=request.target_column,
            num_chunks=request.num_chunks,
            seed=request.seed
        )
        
        metrics = results.get("metrics", {})
        predictions = results.get("predictions")
        
        logger.info(f"Prediction completed. MAPE: {metrics.get('overall_mape', 'N/A')}")
        
        return {
            "message": "Prediction completed successfully",
            "target_column": request.target_column,
            "num_predictions": len(predictions) if predictions else 0,
            "metrics": MetricResponse(
                overall_mape=metrics.get("overall_mape", 0.0),
                chunk_mape=metrics.get("chunk_mape", []),
            ),
            "predictions_shape": list(predictions.shape) if predictions else [0],
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Info endpoint
@app.get(
    "/info",
    tags=["Information"],
    summary="Get API information"
)
async def get_info():
    """Get API information and available endpoints"""
    return {
        "name": "Energy Forecasting API",
        "version": "1.0.0",
        "description": "REST API for time series energy forecasting using Chronos2 model",
        "endpoints": {
            "health": "/health",
            "config": "/config",
            "train": "/train",
            "predict": "/predict",
            "info": "/info",
        },
        "model": "Chronos2",
    }
async def get_status():
    """Get model status"""
    return {
        "model_loaded": model_loaded,
        "model_path": "outputs/model.pt" if model_loaded else None,
        "config_loaded": config_manager is not None,
    }


@app.get("/metrics")
async def get_metrics():
    """Get evaluation metrics"""
    global inference_pipeline
    
    if not model_loaded or inference_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Cannot retrieve metrics."
        )
    
    try:
        metrics_path = "outputs/evaluation_results.json"
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return {"success": True, "metrics": metrics}
        else:
            return {"success": False, "message": "Metrics not found"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
