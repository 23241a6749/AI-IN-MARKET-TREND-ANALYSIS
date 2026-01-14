"""
Production-ready Streamlit app for Multimodal Market Intelligence System.
Deploy this app using: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.multimodal_model import MultimodalPriceForecaster
from src.models.baselines import PriceOnlyLSTM, NaiveMultimodal
from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.training.trainer import MultimodalDataset, MultimodalTrainer
from src.utils.helpers import load_config, split_data
from src.interpretability.attention_viz import AttentionVisualizer
from src.training.evaluator import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Multimodal Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_model_config():
    try:
        return load_config("config/config.yaml")
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


config = load_model_config()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# Sidebar
st.sidebar.title("📊 Market Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📈 Data Overview", "🔮 Predictions", "📊 Model Performance", "⚙️ Settings"]
)

def _get_device_from_config(cfg: dict) -> torch.device:
    """Resolve device similarly to the trainer (CUDA only if available and requested)."""
    requested = (cfg or {}).get("training", {}).get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _checkpoint_path(cfg: dict, model_name: str) -> Path:
    models_dir = Path((cfg or {}).get("paths", {}).get("models_dir", "data/models"))
    return models_dir / f"{model_name}.pt"


# Model loading function (loads architecture from checkpoint config to avoid size mismatch)
@st.cache_resource
def load_trained_model(model_name: str = "multimodal"):
    """Load a pre-trained model checkpoint and rebuild the model from checkpoint config."""
    try:
        base_cfg = load_config()
        ckpt_path = _checkpoint_path(base_cfg, model_name)

        if not ckpt_path.exists():
            return None, None, f"Model checkpoint not found: {ckpt_path}"

        device = _get_device_from_config(base_cfg)
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_cfg = checkpoint.get("config") or base_cfg

        # Build model from checkpoint config so shapes match (e.g., fusion_hidden_size 96 vs 64)
        trainer = MultimodalTrainer(config_dict=ckpt_cfg)
        trainer.create_model(model_type=model_name)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()

        return trainer.model, trainer.device, None
    except Exception as e:
        return None, None, str(e)

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">📊 Multimodal Market Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("### Short-Term Commodity Price Forecasting with Interpretable Multimodal Deep Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This system predicts short-term price movements of agricultural commodities by jointly modeling:
        
        - **📈 Historical Price Dynamics**: Time-series patterns and technical indicators
        - **📰 News Sentiment**: Market-moving information from news articles
        - **🌦️ Weather Conditions**: Supply-side shocks affecting agricultural output
        
        ### Key Features
        
        - 🧠 **Multimodal Deep Learning**: LSTM/Transformer encoders with attention-based fusion
        - 🔍 **Interpretability**: Attention visualizations and ablation studies
        - 📊 **Comprehensive Evaluation**: Multiple baselines and metrics
        - 🎨 **Interactive Dashboard**: Explore predictions and model behavior
        """)
    
    with col2:
        st.markdown("### Quick Stats")
        
        # Check if models exist
        if config:
            models_dir = Path(config['paths']['models_dir'])
            model_files = list(models_dir.glob("*.pt"))
            st.metric("Available Models", len(model_files))
            
            if len(model_files) > 0:
                st.success("✓ Models ready for prediction")
            else:
                st.warning("⚠ No trained models found")
        
        st.markdown("---")
        st.markdown("### Quick Start")
        st.markdown("""
        1. **Data Overview**: Load and explore data
        2. **Predictions**: Make price predictions
        3. **Model Performance**: View model metrics
        """)
    
    st.markdown("---")
    
    # Model status
    if config:
        st.subheader("Model Status")
        col1, col2, col3 = st.columns(3)
        
        models_dir = Path(config['paths']['models_dir'])
        model_names = ['multimodal', 'price_only', 'naive_multimodal']
        
        for i, model_name in enumerate(model_names):
            with [col1, col2, col3][i]:
                model_path = models_dir / f"{model_name}.pt"
                if model_path.exists():
                    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                    st.success(f"✓ {model_name.replace('_', ' ').title()}\n({file_size:.2f} MB)")
                else:
                    st.info(f"○ {model_name.replace('_', ' ').title()}\n(Not trained)")

# Data Overview Page
elif page == "📈 Data Overview":
    st.title("📈 Data Overview")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Data Collection")
        
        commodity = st.selectbox("Commodity", ["ONION", "TOMATO", "POTATO"], index=0)
        location = st.text_input("Location", value=config['data']['location'] if config else "HYDERABAD")
        
        days_back = st.slider("Days to collect", 7, 365, 30)
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        if st.button("🔄 Load/Refresh Data", type="primary"):
            with st.spinner("Collecting data..."):
                try:
                    # Collect data
                    price_collector = PriceCollector(use_yfinance=False)
                    news_collector = NewsCollector()
                    weather_collector = WeatherCollector()
                    
                    weather_location = config['data'].get('location_for_weather', f"{location},IN") if config else f"{location},IN"
                    
                    price_data = price_collector.collect(
                        commodity=commodity,
                        location=location,
                        start_date=start_date
                    )
                    
                    news_data = news_collector.collect(
                        query=f"{commodity.lower()} prices",
                        start_date=start_date
                    )
                    
                    weather_data = weather_collector.collect(
                        location=weather_location,
                        start_date=start_date
                    )
                    
                    # Store in session state
                    st.session_state.price_data = price_data
                    st.session_state.news_data = news_data
                    st.session_state.weather_data = weather_data
                    st.session_state.data_loaded = True
                    
                    st.success(f"✓ Data loaded: {len(price_data)} records")
                except Exception as e:
                    st.error(f"Error collecting data: {e}")
    
    if st.session_state.data_loaded:
        # Price Data
        st.subheader("💰 Price Data")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(st.session_state.price_data.tail(10), width="stretch")
            st.metric("Records", len(st.session_state.price_data))
            st.metric("Date Range", 
                     f"{st.session_state.price_data['date'].min()} to {st.session_state.price_data['date'].max()}")
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.price_data['date'],
                y=st.session_state.price_data['price'],
                mode='lines+markers',
                name='Price',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))
            fig.update_layout(
                title="Price Over Time",
                xaxis_title="Date",
                yaxis_title="Price (INR/quintal)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, width="stretch")
        
        # News Data
        st.subheader("📰 News Sentiment Data")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(st.session_state.news_data.tail(10), width="stretch")
            st.metric("Average Sentiment", f"{st.session_state.news_data['sentiment_score'].mean():.3f}")
            st.metric("Total Articles", int(st.session_state.news_data['article_count'].sum()))
        
        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.news_data['date'],
                    y=st.session_state.news_data['sentiment_score'],
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color='#e74c3c', width=2)
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Bar(
                    x=st.session_state.news_data['date'],
                    y=st.session_state.news_data['article_count'],
                    name='Article Count',
                    marker_color='#95a5a6',
                    opacity=0.6
                ),
                secondary_y=True
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
            fig.update_yaxes(title_text="Article Count", secondary_y=True)
            fig.update_layout(title="News Sentiment Over Time", height=400, hovermode='x unified')
            st.plotly_chart(fig, width="stretch")
        
        # Weather Data
        st.subheader("🌦️ Weather Data")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(st.session_state.weather_data.tail(10), width="stretch")
            st.metric("Avg Temperature", f"{st.session_state.weather_data['temperature_c'].mean():.1f}°C")
            st.metric("Total Rainfall", f"{st.session_state.weather_data['rainfall_mm'].sum():.1f} mm")
        
        with col2:
            fig = make_subplots(rows=3, cols=1, 
                              subplot_titles=("Rainfall (mm)", "Temperature (°C)", "Humidity (%)"),
                              vertical_spacing=0.1)
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['rainfall_mm'],
                    mode='lines+markers',
                    name='Rainfall',
                    line=dict(color='#3498db', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['temperature_c'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#e74c3c', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['humidity_pct'],
                    mode='lines+markers',
                    name='Humidity',
                    line=dict(color='#2ecc71', width=2)
                ),
                row=3, col=1
            )
            
            fig.update_layout(height=600, showlegend=False, hovermode='x unified')
            fig.update_xaxes(title_text="Date", row=3, col=1)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("👆 Click 'Load/Refresh Data' to collect and view data")

# Predictions Page
elif page == "🔮 Predictions":
    st.title("🔮 Price Predictions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠ Please load data first from the Data Overview page.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Model Selection")

            model_name_map = {
                "Multimodal (Attention)": "multimodal",
                "Price Only (Baseline)": "price_only",
                "Naive Multimodal (Baseline)": "naive_multimodal",
            }

            # Show availability and sizes
            if config:
                models_dir = Path(config["paths"]["models_dir"])
            else:
                models_dir = Path("data/models")

            available = {}
            for label, name in model_name_map.items():
                p = models_dir / f"{name}.pt"
                available[label] = p.exists()

            model_choice = st.selectbox(
                "Select Model",
                list(model_name_map.keys()),
                index=0,
                help="Pick any available trained checkpoint. If a model is missing, train it with `python main.py`.",
            )
            selected_model = model_name_map[model_choice]

            for label, ok in available.items():
                name = model_name_map[label]
                p = models_dir / f"{name}.pt"
                if ok:
                    size_mb = p.stat().st_size / (1024 * 1024)
                    st.success(f"✓ {label} ready ({size_mb:.2f} MB)")
                else:
                    st.info(f"○ {label} not found (train first)")

            st.markdown("---")
            if st.button("🔮 Make Predictions", type="primary", width="stretch"):
                with st.spinner("Loading model and making predictions..."):
                    try:
                        # Load model
                        model, device, load_err = load_trained_model(selected_model)
                        if load_err:
                            st.error(f"Could not load **{selected_model}**: {load_err}")
                            st.info("Tip: You can switch models, or train models by running `python main.py` once.")
                            model = None
                        
                        if model is None:
                            st.warning(f"Model **{selected_model}** is not available. Please choose another model or train it first.")
                        else:
                            # Preprocess data
                            preprocessor = DataPreprocessor(
                                lookback_window=config['data']['lookback_window'] if config else 5
                            )
                            
                            merged_df = preprocessor.align_data(
                                st.session_state.price_data,
                                st.session_state.news_data,
                                st.session_state.weather_data
                            )
                            
                            merged_df = FeatureEngineer.engineer_all_features(merged_df)
                            
                            # Create sequences
                            X_price, X_sentiment, X_external, y = preprocessor.create_sequences(merged_df)
                            
                            if len(y) == 0:
                                st.error("Insufficient data to create sequences!")
                            else:
                                # Normalize
                                X_price_norm, X_sentiment_norm, X_external_norm, norm_params = preprocessor.normalize_features(
                                    X_price, X_sentiment, X_external
                                )
                                
                                # Make predictions
                                model.eval()
                                predictions = []
                                probabilities = []
                                attention_weights_list = []
                                
                                # Use last N samples for prediction
                                n_samples = min(50, len(X_price_norm))
                                X_price_batch = torch.FloatTensor(X_price_norm[-n_samples:]).to(device)
                                X_sentiment_batch = torch.FloatTensor(X_sentiment_norm[-n_samples:]).to(device)
                                X_external_batch = torch.FloatTensor(X_external_norm[-n_samples:]).to(device)
                                
                                with torch.no_grad():
                                    if isinstance(model, MultimodalPriceForecaster):
                                        logits, attn = model(X_price_batch, X_sentiment_batch, X_external_batch)
                                        probs = torch.softmax(logits, dim=1)
                                        preds = torch.argmax(probs, dim=1)
                                        attention_weights_list = attn.cpu().numpy()
                                    elif isinstance(model, PriceOnlyLSTM):
                                        logits = model(X_price_batch)
                                        probs = torch.softmax(logits, dim=1)
                                        preds = torch.argmax(probs, dim=1)
                                        attention_weights_list = None
                                    else:  # NaiveMultimodal
                                        logits = model(X_price_batch, X_sentiment_batch, X_external_batch)
                                        probs = torch.softmax(logits, dim=1)
                                        preds = torch.argmax(probs, dim=1)
                                        attention_weights_list = None
                                
                                predictions = preds.cpu().numpy()
                                probabilities = probs.cpu().numpy()
                                
                                # Store in session state
                                st.session_state.predictions = predictions
                                st.session_state.probabilities = probabilities
                                st.session_state.attention_weights = attention_weights_list
                                st.session_state.targets = y[-n_samples:] if len(y) >= n_samples else y
                                st.session_state.model_loaded = True
                                st.session_state.predictions_made = True
                                
                                st.success(f"✓ Predictions made for {n_samples} samples")
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        if st.session_state.predictions_made:
            st.subheader("Prediction Results")
            
            # Metrics
            if 'targets' in st.session_state:
                accuracy = (st.session_state.predictions == st.session_state.targets).mean()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    up_predictions = (st.session_state.predictions == 1).sum()
                    st.metric("Up Predictions", up_predictions)
                with col3:
                    down_predictions = (st.session_state.predictions == 0).sum()
                    st.metric("Down Predictions", down_predictions)
                with col4:
                    avg_confidence = st.session_state.probabilities.max(axis=1).mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            # Prediction table
            st.subheader("Recent Predictions")
            results_df = pd.DataFrame({
                'Prediction': ['📈 Up' if p == 1 else '📉 Down' for p in st.session_state.predictions[-20:]],
                'Confidence': [f"{probs.max():.2%}" for probs in st.session_state.probabilities[-20:]],
                'Up Probability': [f"{probs[1]:.2%}" for probs in st.session_state.probabilities[-20:]],
                'Down Probability': [f"{probs[0]:.2%}" for probs in st.session_state.probabilities[-20:]]
            })
            
            if 'targets' in st.session_state:
                results_df['Actual'] = ['📈 Up' if t == 1 else '📉 Down' for t in st.session_state.targets[-20:]]
                results_df['Correct'] = ['✓' if p == t else '✗' 
                                         for p, t in zip(st.session_state.predictions[-20:], 
                                                        st.session_state.targets[-20:])]
            
            st.dataframe(results_df, width="stretch")
            
            # Prediction distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(x=['Down', 'Up'], 
                          y=[(st.session_state.predictions == 0).sum(), 
                             (st.session_state.predictions == 1).sum()],
                          marker_color=['#e74c3c', '#2ecc71'])
                ])
                fig.update_layout(
                    title="Prediction Distribution",
                    xaxis_title="Direction",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                confidences = st.session_state.probabilities.max(axis=1)
                fig = go.Figure(data=[
                    go.Histogram(x=confidences, nbinsx=20, marker_color='#3498db')
                ])
                fig.update_layout(
                    title="Confidence Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig, width="stretch")
            
            # Attention weights visualization (if available)
            if st.session_state.attention_weights is not None:
                st.subheader("🔍 Attention Weights (Modality Importance)")
                
                modalities = ['Price', 'Sentiment', 'Weather']
                mean_attention = st.session_state.attention_weights.mean(axis=0)
                
                fig = go.Figure(data=[
                    go.Bar(x=modalities, y=mean_attention, 
                          marker_color=['#3498db', '#e74c3c', '#2ecc71'])
                ])
                fig.update_layout(
                    title="Average Modality Importance",
                    xaxis_title="Modality",
                    yaxis_title="Attention Weight",
                    height=400
                )
                st.plotly_chart(fig, width="stretch")

# Model Performance Page
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    
    if not st.session_state.data_loaded:
        st.warning("⚠ Please load data first from the Data Overview page.")
    else:
        st.subheader("Evaluate Models")
        
        if st.button("📊 Evaluate All Models", type="primary"):
            with st.spinner("Evaluating models..."):
                try:
                    preprocessor = DataPreprocessor(
                        lookback_window=config['data']['lookback_window'] if config else 5
                    )
                    
                    merged_df = preprocessor.align_data(
                        st.session_state.price_data,
                        st.session_state.news_data,
                        st.session_state.weather_data
                    )
                    
                    merged_df = FeatureEngineer.engineer_all_features(merged_df)
                    
                    X_price, X_sentiment, X_external, y = preprocessor.create_sequences(merged_df)
                    
                    if len(y) == 0:
                        st.error("Insufficient data to create sequences!")
                    else:
                        X_price_norm, X_sentiment_norm, X_external_norm, _ = preprocessor.normalize_features(
                            X_price, X_sentiment, X_external
                        )
                        
                        _, _, test_data = split_data(
                            X_price_norm, X_sentiment_norm, X_external_norm, y,
                            train_split=0.8, val_split=0.1
                        )
                        
                        test_dataset = MultimodalDataset(*test_data)
                        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
                        
                        evaluator = ModelEvaluator()
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
                        results = {}
                        model_names = ['multimodal', 'price_only', 'naive_multimodal']
                        display_names = ['Multimodal (Attention)', 'Price Only', 'Naive Multimodal']
                        
                        for model_name, display_name in zip(model_names, display_names):
                            model, _, load_err = load_trained_model(model_name)
                            if load_err:
                                st.warning(f"Skipping **{display_name}**: {load_err}")
                            elif model is not None:
                                metrics, _, _, _ = evaluator.evaluate_model(
                                    model, test_loader, device, display_name
                                )
                                results[display_name] = metrics
                        
                        if results:
                            # Display metrics
                            st.subheader("Performance Metrics")
                            
                            metrics_df = pd.DataFrame(results).T
                            # Arrow-safe table: confusion_matrix is a numpy array -> convert to string before display
                            metrics_df_display = metrics_df.copy()
                            if "confusion_matrix" in metrics_df_display.columns:
                                metrics_df_display["confusion_matrix"] = metrics_df_display["confusion_matrix"].apply(
                                    lambda x: np.array(x).tolist() if x is not None else None
                                )
                            st.dataframe(metrics_df_display, width="stretch")
                            
                            # Visualization
                            fig = go.Figure()
                            
                            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                                if metric in metrics_df.columns:
                                    fig.add_trace(go.Bar(
                                        name=metric.replace('_', ' ').title(),
                                        x=metrics_df.index,
                                        y=metrics_df[metric]
                                    ))
                            
                            fig.update_layout(
                                title="Model Comparison",
                                xaxis_title="Model",
                                yaxis_title="Score",
                                barmode='group',
                                height=500
                            )
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.warning("No trained models found for evaluation.")
                            
                except Exception as e:
                    st.error(f"Error evaluating models: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Configuration")
    if config:
        st.json(config)
    else:
        st.error("Configuration not loaded")
    
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**PyTorch Version**: {torch.__version__}")
        st.info(f"**CUDA Available**: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.info(f"**CUDA Device**: {torch.cuda.get_device_name(0)}")
    
    with col2:
        st.info(f"**Python Version**: {sys.version.split()[0]}")
        st.info(f"**Pandas Version**: {pd.__version__}")
        st.info(f"**NumPy Version**: {np.__version__}")

