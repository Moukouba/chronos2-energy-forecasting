"""
Streamlit interface for energy forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import requests
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        padding: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Initialize session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# Sidebar
st.sidebar.title("🔧 Configuration")
api_status = st.sidebar.empty()
config_info = st.sidebar.empty()

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        api_status.success("✅ API Connected")
    else:
        api_status.error("❌ API Unavailable")
except:
    api_status.error("❌ Cannot reach API")

# Main content
st.markdown('<h1 class="main-header">⚡ Energy Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Training", "🔮 Prediction", "📈 Results"])

with tab1:
    st.header("Welcome to Energy Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        This application provides a comprehensive interface for:
        - **Model Training**: Train Chronos2 models on energy data
        - **Forecasting**: Make predictions for multiple energy hubs
        - **Evaluation**: Monitor model performance with detailed metrics
        - **Visualization**: Explore predictions and actuals
        """)
    
    with col2:
        try:
            config_response = requests.get(f"{API_URL}/config", timeout=2)
            if config_response.status_code == 200:
                config_data = config_response.json()
                st.success("Configuration Loaded")
                st.json({
                    "Model": config_data.get("model_name", "N/A"),
                    "Batch Size": config_data.get("batch_size", "N/A"),
                    "Epochs": config_data.get("n_epochs", "N/A"),
                    "Learning Rate": config_data.get("learning_rate", "N/A"),
                })
        except:
            st.warning("Cannot load configuration from API")


with tab2:
    st.header("📊 Model Training")
    
    st.write("Train a new forecasting model on your energy data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_path = st.text_input(
            "Data Path",
            value="/home/moukouba/equilibrium/model_ready.parquet",
            help="Path to training data file"
        )
        output_dir = st.text_input(
            "Output Directory",
            value="outputs",
            help="Where to save trained models"
        )
    
    with col2:
        config_path = st.text_input(
            "Config Path",
            value="config/config.yaml",
            help="Path to configuration file"
        )
        st.info("Training will use the configuration specified in the config file")
    
    if st.button("🚀 Start Training", key="train_btn"):
        with st.spinner("Training in progress..."):
            try:
                response = requests.post(
                    f"{API_URL}/train",
                    json={
                        "data_path": data_path,
                        "output_dir": output_dir,
                        "config_path": config_path
                    },
                    timeout=3600
                )
                
                if response.status_code in [200, 202]:
                    result = response.json()
                    st.markdown(
                        f'<div class="success-box">✅ {result.get("message", "Training completed!")}</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.trained = True
                    
                    if "metrics" in result:
                        st.subheader("Training Metrics")
                        st.json(result["metrics"])
                    
                    st.success(f"Model saved to: {result.get('model_path', 'outputs')}")
                else:
                    st.error(f"Training failed: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Training timed out. The server is still processing.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


with tab3:
    st.header("🔮 Make Predictions")
    
    st.write("Generate forecasts using the trained model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_data_path = st.text_input(
            "Data Path for Prediction",
            value="/home/moukouba/equilibrium/model_ready.parquet",
            help="Path to data for predictions",
            key="pred_data"
        )
    
    with col2:
        num_chunks = st.slider(
            "Number of Prediction Chunks",
            min_value=1,
            max_value=12,
            value=6,
            help="Number of 24-hour chunks to predict"
        )
    
    # Target column selection
    target_col = st.selectbox(
        "Target Column",
        [
            "da_energy_aeci_lmpexpost_ac",
            "da_energy_michigan_hub_lmpexpost_ac",
            "da_energy_minn_hub_lmpexpost_ac"
        ],
        help="Select which energy hub to forecast"
    )
    
    seed = st.number_input("Random Seed (optional)", value=0, step=1, help="Leave 0 for random")
    
    if st.button("🎯 Generate Forecast", key="pred_btn"):
        with st.spinner("Generating predictions..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "data_path": pred_data_path,
                        "target_column": target_col,
                        "num_chunks": num_chunks,
                        "seed": seed if seed > 0 else None
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(
                        f'<div class="success-box">✅ {result.get("message", "Prediction completed!")}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.session_state.predictions = result
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    metrics = result.get("metrics", {})
                    
                    with col1:
                        st.metric(
                            "Overall MAPE",
                            f"{metrics.get('overall_mape', 0):.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Number of Predictions",
                            result.get("num_predictions", 0)
                        )
                    
                    with col3:
                        st.metric(
                            "Target Column",
                            target_col.replace("da_energy_", "").replace("_lmpexpost_ac", "")
                        )
                    
                    # Chunk-wise metrics
                    if metrics.get("chunk_mape"):
                        st.subheader("Chunk-wise MAPE")
                        fig, ax = plt.subplots(figsize=(12, 5))
                        chunk_mape = metrics.get("chunk_mape", [])
                        ax.bar(range(1, len(chunk_mape) + 1), chunk_mape, color='steelblue')
                        ax.set_xlabel("Chunk")
                        ax.set_ylabel("MAPE (%)")
                        ax.set_title(f"MAPE by Chunk - {target_col}")
                        st.pyplot(fig)
                        
                else:
                    st.error(f"Prediction failed: {response.text}")
            except requests.exceptions.Timeout:
                st.error("Prediction request timed out")
            except Exception as e:
                st.error(f"Error: {str(e)}")


with tab4:
    st.header("📈 Results & Visualization")
    
    if st.session_state.predictions:
        st.success("Latest Predictions Available")
        
        result = st.session_state.predictions
        metrics = result.get("metrics", {})
        
        # Display overall metrics
        st.subheader("Overall Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall MAPE", f"{metrics.get('overall_mape', 0):.2f}%")
        with col2:
            st.metric("Total Predictions", result.get("num_predictions", 0))
        with col3:
            st.metric("Timestamp", result.get("timestamp", "N/A"))
        
        # Chunk-wise MAPE visualization
        if metrics.get("chunk_mape"):
            st.subheader("Chunk-wise Performance")
            
            chunk_mape = metrics.get("chunk_mape", [])
            df_chunks = pd.DataFrame({
                "Chunk": [f"Chunk {i+1}" for i in range(len(chunk_mape))],
                "MAPE (%)": chunk_mape
            })
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(df_chunks["Chunk"], df_chunks["MAPE (%)"], color='steelblue', alpha=0.8)
            ax.axhline(y=np.mean(chunk_mape), color='r', linestyle='--', label=f'Average: {np.mean(chunk_mape):.2f}%')
            ax.set_ylabel("MAPE (%)")
            ax.set_xlabel("Chunk")
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
        
        # Display full metrics JSON
        with st.expander("View Full Metrics"):
            st.json(result)
    
    else:
        st.info("No predictions yet. Generate predictions from the Prediction tab to see results.")

# Footer
st.divider()
st.caption(f"Energy Forecasting Dashboard v1.0.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"


def check_api_status():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def load_model_info():
    """Load model information from API."""
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        return response.json()
    except:
        return None


def get_metrics():
    """Get evaluation metrics from API."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def make_prediction(data_path: str, target_col: str, num_chunks: int = 6):
    """Make prediction via API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "data_path": data_path,
                "target_col": target_col,
                "num_chunks": num_chunks,
            },
            timeout=300,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def train_model(data_path: str, output_dir: str = "outputs"):
    """Train model via API."""
    try:
        response = requests.post(
            f"{API_URL}/train",
            json={
                "data_path": data_path,
                "output_dir": output_dir,
            },
            timeout=600,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def plot_forecast(pred_df, actual_df, target_col, chunk_size=24):
    """Plot forecast vs actual."""
    q50_col = [c for c in pred_df.columns if target_col in c and "q0.500" in c][0]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for chunk in range(6):
        start = chunk * chunk_size
        end = (chunk + 1) * chunk_size
        
        ax = axes[chunk]
        ax.plot(actual_df.index[start:end], 
                actual_df[target_col].iloc[start:end], 
                label="Actual", color="black", linewidth=2)
        ax.plot(pred_df.index[start:end], 
                pred_df[q50_col].iloc[start:end], 
                label="Forecast", color="steelblue", linewidth=2, linestyle="--")
        
        ax.set_title(f"Chunk {chunk + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy (MWh)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    st.markdown('<p class="main-header">⚡ Energy Forecasting Dashboard</p>', 
                unsafe_allow_html=True)
    
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select Page", ["Home", "Predictions", "Training", "Metrics"])
        
        st.divider()
        st.header("Configuration")
        
        # Check API status
        if check_api_status():
            st.success("✅ API is running")
            st.session_state.api_status = "running"
        else:
            st.warning("⚠️ API is not running")
            st.session_state.api_status = "stopped"
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Main content
    if page == "Home":
        st.header("Welcome to Energy Forecasting Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", 
                     "✅ Running" if st.session_state.api_status == "running" else "⚠️ Stopped")
        
        with col2:
            st.metric("Model Status", 
                     "✅ Loaded" if st.session_state.model_loaded else "❌ Not Loaded")
        
        with col3:
            st.metric("Version", "1.0.0")
        
        st.divider()
        
        st.subheader("Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📊 **Predictions**: Make time series forecasts using trained model")
        
        with col2:
            st.info("🚀 **Training**: Retrain model with new data")
        
        st.info("📈 **Metrics**: View evaluation metrics and performance")
        
        st.divider()
        
        st.subheader("Quick Start")
        st.code("""
        # Start the API server
        cd energy_forecast
        python -m uvicorn src.api.api:app --host 0.0.0.0 --port 8000
        
        # Start the Streamlit app
        streamlit run app.py
        """, language="bash")
    
    elif page == "Predictions":
        st.header("Make Predictions")
        
        if st.session_state.api_status != "running":
            st.error("API is not running. Please start the API server first.")
            st.stop()
        
        # Load model info
        model_info = load_model_info()
        if model_info:
            st.success(f"Model loaded: {model_info.get('name', 'Energy Forecasting Model')}")
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            data_path = st.text_input(
                "Data Path",
                value="/home/moukouba/equilibrium/model_ready.parquet"
            )
        
        with col2:
            target_col = st.selectbox(
                "Target Column",
                options=[
                    "da_energy_aeci_lmpexpost_ac",
                    "da_energy_michigan_hub_lmpexpost_ac",
                    "da_energy_minn_hub_lmpexpost_ac",
                ],
                index=0
            )
        
        num_chunks = st.slider("Number of chunks", 1, 10, 6)
        
        if st.button("🚀 Make Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                result = make_prediction(data_path, target_col, num_chunks)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(result.get("message", "Predictions completed!"))
                    
                    if result.get("predictions_available"):
                        # Display metrics
                        metrics = result.get("metrics", {})
                        st.subheader("Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Overall MAPE", 
                                    f"{metrics.get('overall_mape', 0):.2f}%")
                        
                        with col2:
                            st.metric("Chunk 1 MAPE", 
                                    f"{metrics.get('chunk_mape', [0]*6)[0]:.2f}%")
                        
                        with col3:
                            st.metric("Chunk 6 MAPE", 
                                    f"{metrics.get('chunk_mape', [0]*6)[5]:.2f}%")
    
    elif page == "Training":
        st.header("Train Model")
        
        if st.session_state.api_status != "running":
            st.error("API is not running. Please start the API server first.")
            st.stop()
        
        st.info("Training will run in the background. You can continue using the dashboard.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_path = st.text_input(
                "Training Data Path",
                value="/home/moukouba/equilibrium/model_ready.parquet"
            )
        
        with col2:
            output_dir = st.text_input("Output Directory", value="outputs")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model..."):
                result = train_model(data_path, output_dir)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(result.get("message", "Training completed!"))
                    
                    if result.get("model_path"):
                        st.info(f"Model saved to: {result['model_path']}")
                        st.info(f"ONNX model saved to: {result['onnx_path']}")
                        
                        st.session_state.model_loaded = True
                        st.rerun()
    
    elif page == "Metrics":
        st.header("Evaluation Metrics")
        
        if st.session_state.api_status != "running":
            st.error("API is not running. Please start the API server first.")
            st.stop()
        
        metrics_data = get_metrics()
        
        if metrics_data and metrics_data.get("success"):
            metrics = metrics_data.get("metrics", {})
            
            if metrics:
                # Create summary table
                summary_data = []
                for hub, data in metrics.items():
                    row = {
                        "Hub": hub.replace("da_energy_", "").replace("_lmpexpost_ac", ""),
                        "Overall MAPE": f"{data.get('overall_mape', 0):.2f}%",
                    }
                    for i, chunk_mape in enumerate(data.get("chunk_mape", [])):
                        row[f"Chunk {i+1}"] = f"{chunk_mape:.2f}%"
                    summary_data.append(row)
                
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Plot metrics
                st.subheader("MAPE Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(df))
                width = 0.15
                
                # Extract chunk values for plotting
                chunk_cols = [col for col in df.columns if col.startswith("Chunk")]
                
                if chunk_cols:
                    for i, chunk in enumerate(chunk_cols):
                        values = [float(df[chunk][j].replace("%", "")) for j in range(len(df))]
                        ax.bar(x + i*width, values, width, label=chunk)
                
                ax.set_xlabel("Hub")
                ax.set_ylabel("MAPE (%)")
                ax.set_title("MAPE by Hub and Chunk")
                ax.set_xticks(x + width * 2.5)
                ax.set_xticklabels(df["Hub"])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download metrics
                if st.button("📥 Download Metrics"):
                    metrics_json = json.dumps(metrics_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=metrics_json,
                        file_name="evaluation_metrics.json",
                        mime="application/json",
                    )
            else:
                st.info("No metrics available. Please train the model first.")
        else:
            st.info("No metrics available. Please train the model first.")


if __name__ == "__main__":
    main()
