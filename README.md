# 📊 Multimodal Market Intelligence System  
### Short-Term Commodity Price Forecasting using Multimodal Deep Learning

🔗 **Live Demo**  
https://23241a6749-ai-in-market-trend-analysis-app-1ystbp.streamlit.app/

🔗 **GitHub Repository**  
https://github.com/23241a6749/AI-IN-MARKET-TREND-ANALYSIS

---

## 📌 Project Overview

This project presents an end-to-end Multimodal Market Intelligence System designed to predict short-term agricultural commodity price movement (Up/Down) by jointly modeling multiple real-world data sources such as historical prices, news sentiment, and weather conditions. The system uses a multimodal deep learning architecture with attention-based fusion, allowing it to dynamically learn the importance of each data source while also providing interpretable predictions. The trained models are deployed as an interactive Streamlit web application.

---

## ✅ Project Status

✔ Core system implemented  
✔ Models trained and evaluated  
✔ Interpretability analysis completed  
✔ Streamlit application deployed  
✔ Ready for academic submission and demonstration  

---

## 🧠 Key Features

- Price Movement Prediction: Next-day price direction (Up/Down)  
- Multimodal Deep Learning: Price, news sentiment, and weather signals  
- Attention-Based Fusion: Learns modality importance dynamically  
- Baseline Comparisons: Price-only and naive multimodal models  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- Interpretability: Attention weight visualization and ablation studies  
- Interactive Dashboard: Streamlit-based user interface  
- Cloud Deployment: Streamlit Community Cloud  

---

## 🏗️ Project Structure

AI-IN-MARKET-TREND-ANALYSIS/
│
├── app.py                     # Streamlit application (production-ready)
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml            # Model and data configuration
│
├── src/
│   ├── data/
│   │   ├── collectors.py      # Price, news, weather collection
│   │   ├── preprocessor.py    # Data alignment & sequence creation
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── multimodal_model.py # Attention-based multimodal model
│   │   ├── baselines.py        # Baseline models
│   │   └── attention.py
│   │
│   ├── training/
│   │   ├── trainer.py          # Training pipeline
│   │   └── evaluator.py        # Evaluation logic
│   │
│   ├── interpretability/
│   │   ├── attention_viz.py    # Attention visualization
│   │   └── ablation.py         # Ablation studies
│   │
│   └── utils/
│       └── helpers.py
│
├── data/
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Processed datasets
│   └── models/                # Trained model checkpoints (.pt)
│
└── README.md

---

## 🔄 System Workflow

1. Data Collection  
   - Commodity price data  
   - News sentiment scores  
   - Weather parameters (temperature, humidity, rainfall)  

2. Data Preprocessing  
   - Date-wise alignment  
   - Missing value handling  
   - Feature normalization  

3. Sequence Generation  
   - Sliding window time-series sequences  

4. Model Training  
   - Multimodal attention-based model  
   - Baseline models for comparison  

5. Evaluation & Interpretability  
   - Performance metrics  
   - Attention visualization  
   - Ablation analysis  

6. Deployment  
   - Interactive Streamlit dashboard  

---

## 🧠 Model Architecture

The system uses a three-stream neural network architecture:

- Price Encoder: Processes historical price sequences  
- Sentiment Encoder: Processes news sentiment features  
- Weather Encoder: Processes weather signals  
- Attention Fusion Layer: Dynamically weights each modality  
- Prediction Head: Binary classification (Up / Down)  

This architecture improves both prediction accuracy and model interpretability.

---

## 📊 Evaluation & Results

The proposed model was evaluated against baseline approaches including a price-only model and a naive multimodal model without attention. Standard classification metrics such as accuracy, precision, recall, F1-score, and confusion matrices were used for evaluation.

Key observations:
- The multimodal attention-based model outperformed baseline models  
- Weather data had the strongest influence on prediction accuracy  
- Attention weights provided meaningful explanations of model behavior  

---

## 🚀 Running the Project Locally

git clone https://github.com/23241a6749/AI-IN-MARKET-TREND-ANALYSIS.git  
cd AI-IN-MARKET-TREND-ANALYSIS  
pip install -r requirements.txt  
streamlit run app.py  

The application will be available at http://localhost:8501.

---

## 🌐 Deployment

The project is deployed using Streamlit Community Cloud.

Live Application:  
https://23241a6749-ai-in-market-trend-analysis-app-1ystbp.streamlit.app/

---

## 🧪 AI Usage Disclosure

AI tools (ChatGPT) were used during development for debugging support, concept clarification, and documentation assistance. All model design, implementation, training, evaluation, and deployment were performed and verified by the author.

---

## 📈 Future Improvements

- Extend to regression-based price forecasting  
- Support multi-day prediction horizons  
- Incorporate additional external data sources  
- Improve scalability and real-time data ingestion  

---

## 👤 Author

Developed as part of IITR Module E academic project.
