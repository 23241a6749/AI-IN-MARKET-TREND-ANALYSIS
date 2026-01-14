# 📊 Multimodal Market Intelligence System

**Short-Term Commodity Price Forecasting with Interpretable Multimodal Deep Learning**

## 📚 Complete Documentation

- **[PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md)** - Complete project overview and status
- **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** - Detailed technical reference
- **[CHALLENGES_AND_SOLUTIONS.md](CHALLENGES_AND_SOLUTIONS.md)** - All challenges faced and solutions
- **[COMPLETE_PROJECT_DOCUMENTATION.md](COMPLETE_PROJECT_DOCUMENTATION.md)** - Comprehensive documentation

## ✅ Project Status: COMPLETE

**All components implemented, tested, and production-ready!**

## Project Overview

This project develops an end-to-end Multimodal Market Intelligence System that predicts short-term price movements of agricultural commodities (configurable location, default: Nashik, India) by jointly modeling:

- **Historical price dynamics** (time-series patterns)
- **News sentiment** (market-moving information)
- **Weather conditions** (supply-side shocks)

The system uses a three-stream deep learning architecture with attention-based multimodal fusion, enabling dynamic learning of signal importance and providing interpretable predictions.

## Features

- 🧠 **Flexible Deep Learning**: Choose between LSTM, GRU, or Transformer encoders with attention-based fusion
- 📍 **Configurable Location**: Change location from Nashik to any city/region
- 📈 **Price Forecasting**: Next-day price direction prediction (extendable to regression)
- 📰 **News Sentiment Analysis**: Automated sentiment extraction from market news
- 🌦️ **Weather Integration**: External contextual signals for supply-side analysis
- 🔍 **Interpretability**: Attention visualizations and ablation studies
- 🎨 **Interactive Dashboard**: Streamlit web application for exploration
- 📊 **Comprehensive Evaluation**: Multiple baselines and metrics
- ✅ **Data Validation**: Automatic data quality checks and cleaning
- 📝 **Comprehensive Logging**: Detailed logging system with file and console output
- 🔌 **API Integration**: Support for real data APIs (OpenWeatherMap, NewsAPI)
- 🎯 **Enhanced Architecture**: Batch normalization and improved weight initialization

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors.py      # Data collection (price, news, weather)
│   │   ├── preprocessor.py    # Data cleaning and alignment
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal_model.py # Main multimodal architecture
│   │   ├── baselines.py        # Baseline models
│   │   └── attention.py        # Attention mechanisms
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training pipeline
│   │   └── evaluator.py        # Evaluation metrics
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── attention_viz.py    # Attention visualizations
│   │   └── ablation.py         # Ablation studies
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── data/
│   ├── raw/                    # Raw collected data
│   ├── processed/              # Processed datasets
│   └── models/                 # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── config/
│   └── config.yaml             # Configuration file
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository** (or navigate to project directory)

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional, for real data APIs):
```bash
# Create .env file in project root
WEATHER_API_KEY=your_openweathermap_api_key
NEWS_API_KEY=your_newsapi_key
```

   Get API keys:
   - Weather: https://openweathermap.org/api (free tier available)
   - News: https://newsapi.org/ (free tier available)
   
   **📖 See [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) for detailed setup instructions**

## Quick Start

### Option 1: Run Complete Pipeline

Execute the full pipeline (data collection, training, evaluation):

```bash
python main.py
```

This will:
- Collect price, news, and weather data
- Preprocess and engineer features
- Train multimodal model and baselines
- Evaluate and compare models
- Generate interpretability visualizations
- Run ablation studies

### Option 2: Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard/app.py
```

Or use the production-ready app:

```bash
streamlit run app.py
```

The dashboard provides:
- Data visualization and exploration
- Interactive model training
- Real-time predictions
- Attention weight visualizations
- Model comparison tools

## 🚀 Deployment

### Streamlit Cloud Deployment

The app is ready for deployment on Streamlit Cloud! See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed instructions.

**Quick deployment steps:**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Set main file to `app.py`
   - Click "Deploy"

3. **Add API keys (optional):**
   - In Streamlit Cloud settings, add secrets:
     ```
     NEWS_API_KEY=your_key
     WEATHER_API_KEY=your_key
     ```

Your app will be live at `https://your-app-name.streamlit.app`!

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Usage

### 1. Data Collection

```python
from src.data.collectors import PriceCollector, NewsCollector, WeatherCollector

# Collect price data
price_collector = PriceCollector()
price_data = price_collector.collect("ONION", "NASHIK", start_date="2020-01-01")

# Collect news data
news_collector = NewsCollector()
news_data = news_collector.collect("onion prices", start_date="2020-01-01")

# Collect weather data
weather_collector = WeatherCollector()
weather_data = weather_collector.collect("Nashik", start_date="2020-01-01")
```

### 2. Training the Model

```python
from src.training.trainer import MultimodalTrainer
from torch.utils.data import DataLoader

# Create trainer
trainer = MultimodalTrainer(config_path="config/config.yaml")

# Create model
trainer.create_model(model_type='multimodal')

# Train
history = trainer.train(train_loader, val_loader, model_type='multimodal')
```

### 3. Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate through the pages:
- **Home**: Overview and quick start
- **Data Overview**: Explore collected data
- **Model Training**: Train models interactively
- **Predictions**: Make and visualize predictions
- **Interpretability**: View attention weights and model behavior
- **Model Comparison**: Compare different models

## Model Architecture

The system uses a three-stream architecture:

1. **Price Encoder**: LSTM network processing historical price sequences
2. **Sentiment Encoder**: LSTM network processing news sentiment sequences
3. **External Signal Encoder**: Lightweight encoder for weather data
4. **Attention Fusion**: Dynamically weights modalities based on relevance
5. **Prediction Head**: Final classification/regression layers

## Evaluation

The system is evaluated against:
- Price-only LSTM baseline
- Naïve multimodal (concatenation) baseline
- Metrics: RMSE, MAE, Directional Accuracy

## Results & Interpretability

- **Attention Visualizations**: Heatmaps showing modality importance over time
- **Ablation Studies**: Quantifying contribution of each modality
- **Error Analysis**: Identifying failure modes and success patterns

## Recent Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed information about recent enhancements:

- ✅ **Real Data Integration**: NewsAPI, OpenWeatherMap, and yfinance support
- ✅ **FinBERT Sentiment Analysis**: Advanced financial sentiment model
- ✅ **Comprehensive Logging System**: Detailed logging with file and console output
- ✅ **Data Validation**: Automatic data quality checks and cleaning
- ✅ **Enhanced Model Architecture**: Batch normalization and improved initialization
- ✅ **Improved Error Handling**: Graceful fallbacks and error recovery

**🚀 New: Real Data Integration!** See [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) to set up real APIs.

## Future Extensions

- Multi-day forecasting horizons
- Probabilistic forecasting
- Additional external signals (yield data, trade volumes)
- Regional news source integration
- Unit test suite expansion
- Hyperparameter tuning automation
- Model ensembling

## License

MIT

## Author

Developed as part of IITR Module E project.

