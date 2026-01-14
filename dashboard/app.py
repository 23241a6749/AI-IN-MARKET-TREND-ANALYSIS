"""
Streamlit dashboard for Multimodal Market Intelligence System.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.multimodal_model import MultimodalPriceForecaster
from src.models.baselines import PriceOnlyLSTM, NaiveMultimodal
from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.training.trainer import MultimodalDataset
from src.utils.helpers import load_config, split_data
from src.interpretability.attention_viz import AttentionVisualizer

# Page configuration
st.set_page_config(
    page_title="Multimodal Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_model_config():
    return load_config()

config = load_model_config()

# Sidebar
st.sidebar.title("📊 Market Intelligence System")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "📈 Data Overview", "🤖 Model Training", "🔮 Predictions", "🔍 Interpretability", "📊 Model Comparison"]
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Home Page
if page == "🏠 Home":
    st.title("📊 Multimodal Market Intelligence System")
    st.markdown("### Short-Term Commodity Price Forecasting with Interpretable Multimodal Deep Learning")
    
    st.markdown("""
    This system predicts short-term price movements of agricultural commodities by jointly modeling:
    
    - **📈 Historical Price Dynamics**: Time-series patterns and technical indicators
    - **📰 News Sentiment**: Market-moving information from news articles
    - **🌦️ Weather Conditions**: Supply-side shocks affecting agricultural output
    
    ### Key Features
    
    - 🧠 **Multimodal Deep Learning**: LSTM encoders with attention-based fusion
    - 🔍 **Interpretability**: Attention visualizations and ablation studies
    - 📊 **Comprehensive Evaluation**: Multiple baselines and metrics
    - 🎨 **Interactive Dashboard**: Explore predictions and model behavior
    """)
    
    st.markdown("---")
    st.markdown("### Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Data Overview**
        - View collected price, news, and weather data
        - Explore data statistics and visualizations
        """)
    
    with col2:
        st.markdown("""
        **2. Model Training**
        - Train multimodal model and baselines
        - Monitor training progress
        """)
    
    with col3:
        st.markdown("""
        **3. Predictions & Analysis**
        - Make predictions on new data
        - Explore interpretability features
        """)

# Data Overview Page
elif page == "📈 Data Overview":
    st.title("📈 Data Overview")
    
    if st.button("Load/Refresh Data"):
        with st.spinner("Collecting data..."):
            # Collect data
            price_collector = PriceCollector()
            news_collector = NewsCollector()
            weather_collector = WeatherCollector()
            
            price_data = price_collector.collect(
                commodity="ONION",
                location=config['data']['location'],
                start_date="2020-01-01"
            )
            
            news_data = news_collector.collect(
                query="onion prices",
                start_date="2020-01-01"
            )
            
            weather_data = weather_collector.collect(
                location=config['data'].get('location_for_weather', f"{config['data']['location']},IN"),
                start_date="2020-01-01"
            )
            
            # Store in session state
            st.session_state.price_data = price_data
            st.session_state.news_data = news_data
            st.session_state.weather_data = weather_data
            st.session_state.data_loaded = True
            
            st.success("Data loaded successfully!")
    
    if st.session_state.data_loaded:
        # Price Data
        st.subheader("💰 Price Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(st.session_state.price_data.tail(10))
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.price_data['date'],
                y=st.session_state.price_data['price'],
                mode='lines',
                name='Price',
                line=dict(color='#3498db', width=2)
            ))
            fig.update_layout(
                title="Price Over Time",
                xaxis_title="Date",
                yaxis_title="Price (INR/quintal)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # News Data
        st.subheader("📰 News Sentiment Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(st.session_state.news_data.tail(10))
        
        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.news_data['date'],
                    y=st.session_state.news_data['sentiment_score'],
                    mode='lines',
                    name='Sentiment Score',
                    line=dict(color='#e74c3c')
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Bar(
                    x=st.session_state.news_data['date'],
                    y=st.session_state.news_data['article_count'],
                    name='Article Count',
                    marker_color='#95a5a6',
                    opacity=0.5
                ),
                secondary_y=True
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
            fig.update_yaxes(title_text="Article Count", secondary_y=True)
            fig.update_layout(title="News Sentiment Over Time", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather Data
        st.subheader("🌦️ Weather Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(st.session_state.weather_data.tail(10))
        
        with col2:
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Rainfall", "Temperature", "Humidity"))
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['rainfall_mm'],
                    mode='lines',
                    name='Rainfall',
                    line=dict(color='#3498db')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['temperature_c'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='#e74c3c')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.weather_data['date'],
                    y=st.session_state.weather_data['humidity_pct'],
                    mode='lines',
                    name='Humidity',
                    line=dict(color='#2ecc71')
                ),
                row=3, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Data Overview page.")
    else:
        model_type = st.selectbox(
            "Select Model Type",
            ["Multimodal (Attention)", "Price Only (Baseline)", "Naive Multimodal (Baseline)"]
        )
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Preprocess data
                preprocessor = DataPreprocessor(lookback_window=30)
                merged_df = preprocessor.align_data(
                    st.session_state.price_data,
                    st.session_state.news_data,
                    st.session_state.weather_data
                )
                
                # Feature engineering
                merged_df = FeatureEngineer.engineer_all_features(merged_df)
                
                # Create sequences
                X_price, X_sentiment, X_external, y = preprocessor.create_sequences(merged_df)
                
                # Normalize
                X_price_norm, X_sentiment_norm, X_external_norm, norm_params = preprocessor.normalize_features(
                    X_price, X_sentiment, X_external
                )
                
                # Split data
                train_data, val_data, test_data = split_data(
                    X_price_norm, X_sentiment_norm, X_external_norm, y
                )
                
                # Create datasets
                train_dataset = MultimodalDataset(*train_data)
                val_dataset = MultimodalDataset(*val_data)
                test_dataset = MultimodalDataset(*test_data)
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Train model
                from src.training.trainer import MultimodalTrainer
                trainer = MultimodalTrainer()
                
                model_name_map = {
                    "Multimodal (Attention)": "multimodal",
                    "Price Only (Baseline)": "price_only",
                    "Naive Multimodal (Baseline)": "naive_multimodal"
                }
                
                history = trainer.train(
                    train_loader,
                    val_loader,
                    model_type=model_name_map[model_type],
                    model_name=model_name_map[model_type]
                )
                
                st.session_state.model = trainer.model
                st.session_state.model_type = model_type
                st.session_state.train_history = history
                st.session_state.test_loader = test_loader
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                
                # Plot training history
                st.subheader("Training History")
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                
                epochs = list(range(1, len(history['train_loss']) + 1))
                
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='#3498db')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='#e74c3c')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', line=dict(color='#3498db')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', line=dict(color='#e74c3c')),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

# Predictions Page
elif page == "🔮 Predictions":
    st.title("🔮 Predictions")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first from the Model Training page.")
    else:
        st.subheader("Make Predictions")
        
        # Get a batch from test set
        if st.button("Generate Predictions"):
            model = st.session_state.model
            test_loader = st.session_state.test_loader
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model.eval()
            predictions = []
            targets = []
            attention_weights_list = []
            
            with torch.no_grad():
                for batch in list(test_loader)[:5]:  # First 5 batches
                    price_seq, sentiment_seq, external_seq, target = batch
                    price_seq = price_seq.to(device)
                    sentiment_seq = sentiment_seq.to(device)
                    external_seq = external_seq.to(device)
                    
                    if hasattr(model, 'predict'):
                        pred, prob, attn = model.predict(price_seq, sentiment_seq, external_seq)
                    else:
                        logits, attn = model(price_seq, sentiment_seq, external_seq)
                        prob = torch.softmax(logits, dim=1)
                        pred = torch.argmax(prob, dim=1)
                    
                    predictions.extend(pred.cpu().numpy())
                    targets.extend(target.cpu().numpy())
                    attention_weights_list.extend(attn.cpu().numpy())
            
            st.session_state.predictions = np.array(predictions)
            st.session_state.targets = np.array(targets)
            st.session_state.attention_weights = np.array(attention_weights_list)
        
        if 'predictions' in st.session_state:
            # Display predictions
            results_df = pd.DataFrame({
                'Prediction': ['Up' if p == 1 else 'Down' for p in st.session_state.predictions],
                'Actual': ['Up' if t == 1 else 'Down' for t in st.session_state.targets],
                'Correct': st.session_state.predictions == st.session_state.targets
            })
            
            st.dataframe(results_df)
            
            # Accuracy
            accuracy = (st.session_state.predictions == st.session_state.targets).mean()
            st.metric("Prediction Accuracy", f"{accuracy:.2%}")

# Interpretability Page
elif page == "🔍 Interpretability":
    st.title("🔍 Model Interpretability")
    
    if not st.session_state.model_trained or 'attention_weights' not in st.session_state:
        st.warning("Please generate predictions first from the Predictions page.")
    else:
        st.subheader("Attention Weight Visualizations")
        
        attention_weights = st.session_state.attention_weights
        
        # Modality importance
        modalities = ['Price', 'Sentiment', 'Weather']
        mean_attention = attention_weights.mean(axis=0)
        
        fig = go.Figure(data=[
            go.Bar(x=modalities, y=mean_attention, marker_color=['#3498db', '#e74c3c', '#2ecc71'])
        ])
        fig.update_layout(
            title="Average Modality Importance",
            xaxis_title="Modality",
            yaxis_title="Attention Weight",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Attention over time
        fig = go.Figure()
        n_samples = min(50, len(attention_weights))
        x_axis = list(range(n_samples))
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=attention_weights[:n_samples, 0],
            mode='lines',
            name='Price',
            stackgroup='one',
            fillcolor='#3498db'
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=attention_weights[:n_samples, 1],
            mode='lines',
            name='Sentiment',
            stackgroup='one',
            fillcolor='#e74c3c'
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=attention_weights[:n_samples, 2],
            mode='lines',
            name='Weather',
            stackgroup='one',
            fillcolor='#2ecc71'
        ))
        
        fig.update_layout(
            title="Attention Weights Over Time",
            xaxis_title="Sample",
            yaxis_title="Attention Weight",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Model Comparison Page
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    
    st.info("Train multiple models from the Model Training page to compare their performance.")
    
    # Placeholder for model comparison
    st.markdown("""
    ### Comparison Metrics
    
    Compare models based on:
    - **Accuracy**: Overall prediction accuracy
    - **Precision**: True positive rate
    - **Recall**: Sensitivity
    - **F1 Score**: Harmonic mean of precision and recall
    
    ### Baseline Models
    
    1. **Price Only LSTM**: Uses only historical price data
    2. **Naive Multimodal**: Simple concatenation of all modalities
    3. **Multimodal with Attention**: Our proposed model with attention-based fusion
    """)

