#!/usr/bin/env python3
"""
Main entry point for energy forecasting pipeline
"""

import argparse
import sys
import logging
from pathlib import Path

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.inference_pipeline import InferencePipeline
from src.config.configuration import ConfigManager
from src.utils.common import (
    ensure_dir,
    load_json,
    save_json,
    setup_logging,
    print_section,
    calculate_metrics
)

# Setup logging
logger = setup_logging()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Energy Forecasting Pipeline - Production Ready Time Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train --data-path /path/to/data.parquet
  
  # Make predictions
  python main.py predict --data-path /path/to/data.parquet --target-col da_energy_aeci_lmpexpost_ac
  
  # Run API server
  python main.py api --host 0.0.0.0 --port 8000
  
  # Run Streamlit dashboard
  python main.py dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the forecasting model")
    train_parser.add_argument(
        "--data-path",
        type=str,
        default="/home/moukouba/equilibrium/model_ready.parquet",
        help="Path to training data"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for model and results"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make forecasting predictions")
    predict_parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to data for prediction"
    )
    predict_parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Target column to predict"
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/model.pt",
        help="Path to trained model"
    )
    predict_parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    predict_parser.add_argument(
        "--num-chunks",
        type=int,
        default=6,
        help="Number of chunks for rolling prediction"
    )
    predict_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port"
    )
    api_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run Streamlit dashboard")
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port"
    )
    
    return parser.parse_args()


def train_model(args):
    """Train the model."""
    print_section("TRAINING ENERGY FORECASTING MODEL")
    
    try:
        # Validate input
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            sys.exit(1)
        
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        logger.info(f"Starting training with data: {args.data_path}")
        
        # Create and run pipeline
        pipeline = TrainingPipeline(config)
        pipeline.setup()
        
        results = pipeline.run(
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        print_section("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Model saved to: {results.get('model_path', 'outputs')}")
        
        # Display results
        if "metrics" in results:
            logger.info("Training metrics:")
            for key, value in results["metrics"].items():
                logger.info(f"  {key}: {value}")
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


def make_predictions(args):
    """Make predictions with trained model."""
    print_section("MAKING PREDICTIONS")
    
    try:
        # Validate input
        if not Path(args.data_path).exists():
            logger.error(f"Data file not found: {args.data_path}")
            sys.exit(1)
        
        if not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            sys.exit(1)
        
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        logger.info(f"Loading model from {args.model_path}")
        inference_pipeline = InferencePipeline(
            model_path=args.model_path,
            config=config
        )
        inference_pipeline.load_model()
        
        logger.info(f"Making predictions for {args.target_col}")
        results = inference_pipeline.predict(
            data_path=args.data_path,
            target_col=args.target_col,
            num_chunks=args.num_chunks,
            seed=args.seed,
        )
        
        print_section("PREDICTION RESULTS")
        
        metrics = results.get("metrics", {})
        logger.info(f"Overall MAPE: {metrics.get('overall_mape', 'N/A'):.2f}%")
        
        if metrics.get("chunk_mape"):
            logger.info("Chunk-wise MAPE:")
            for i, mape in enumerate(metrics.get("chunk_mape", []), 1):
                logger.info(f"  Chunk {i}: {mape:.2f}%")
        
        return results
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


def run_api(args):
    """Run FastAPI server."""
    try:
        import uvicorn
        logger.info(f"Starting FastAPI server at {args.host}:{args.port}")
        
        uvicorn.run(
            "src.api.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level=logging.getLevelName(logging.INFO).lower()
        )
    
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        sys.exit(1)


def run_dashboard(args):
    """Run Streamlit dashboard."""
    try:
        import subprocess
        logger.info(f"Starting Streamlit dashboard on port {args.port}")
        
        subprocess.run(
            ["streamlit", "run", "app.py", "--server.port", str(args.port)],
            check=True
        )
    
    except FileNotFoundError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start dashboard: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "predict":
        make_predictions(args)
    elif args.command == "api":
        run_api(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    else:
        print("Usage: python main.py [train|predict|api|dashboard] --help")
        sys.exit(1)


if __name__ == "__main__":
    main()
