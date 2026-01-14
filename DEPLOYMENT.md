# Streamlit Deployment Guide

This guide will help you deploy the Multimodal Market Intelligence System using Streamlit.

## Quick Start

### Local Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   - Open your browser to `http://localhost:8501`

## Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. **Ensure your repository structure:**
   ```
   your-repo/
   ├── app.py                    # Main Streamlit app
   ├── requirements.txt          # Python dependencies
   ├── config/
   │   └── config.yaml          # Configuration file
   ├── data/
   │   └── models/              # Trained models (optional)
   ├── src/                     # Source code
   └── .streamlit/
       └── config.toml          # Streamlit config
   ```

2. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
   - Sign up/login with your GitHub account

2. **Click "New app"**

3. **Configure your app:**
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
   - **Python version**: 3.11 (recommended)

4. **Advanced Settings (Optional):**
   - **Secrets**: Add environment variables if needed:
     ```
     NEWS_API_KEY=your_news_api_key
     WEATHER_API_KEY=your_weather_api_key
     ```

5. **Click "Deploy"**

### Step 3: Access Your Deployed App

- Streamlit Cloud will provide you with a URL like: `https://your-app-name.streamlit.app`
- Your app will automatically redeploy when you push changes to the main branch

## Environment Variables

If you're using API keys for data collection, add them as secrets in Streamlit Cloud:

1. Go to your app settings in Streamlit Cloud
2. Click "Secrets"
3. Add your keys:
   ```toml
   NEWS_API_KEY = "your_news_api_key_here"
   WEATHER_API_KEY = "your_weather_api_key_here"
   ```

## Troubleshooting

### Common Issues

1. **Models not found:**
   - Ensure trained models are in `data/models/` directory
   - Or train models first using `python main.py`

2. **Import errors:**
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version matches (3.11 recommended)

3. **Data collection fails:**
   - Check API keys are set correctly
   - The app will fall back to synthetic data if APIs are unavailable

4. **Memory issues:**
   - Streamlit Cloud has memory limits
   - Consider using smaller models or reducing batch sizes

## Local Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Development Tips

- Use `st.cache_resource` for model loading (already implemented)
- Use `st.cache_data` for data loading (already implemented)
- Test with small datasets first
- Check logs in the terminal for errors

## Features

The deployed app includes:

- 📈 **Data Overview**: View price, news, and weather data
- 🔮 **Predictions**: Make price predictions using trained models
- 📊 **Model Performance**: Evaluate and compare models
- ⚙️ **Settings**: View configuration and system info

## Notes

- Models need to be trained before making predictions
- The app will work with synthetic data if APIs are not configured
- All visualizations use Plotly for interactivity
- The app is responsive and works on mobile devices

## Support

For issues or questions:
1. Check the logs in Streamlit Cloud
2. Review the main README.md
3. Check configuration in `config/config.yaml`

