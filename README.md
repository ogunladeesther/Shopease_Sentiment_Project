# Shopease_Sentiment_Project

## Overview
A full-scale sentiment analysis pipeline designed for the ShopEase e-commerce platform. The system analyzes customer reviews and classifies them as positive, negative, or neutral using machine learning and natural language processing techniques.

## Features
- Multilingual sentiment analysis supporting reviews in English, French, German, Spanish, and more
- REST API built with FastAPI for single and batch predictions
- Interactive dashboard built with Streamlit for visualizing sentiment results
- Model tracking and experiment logging with MLflow and DagsHub
- Automated model retraining pipeline

## Tech Stack
- **Model:** DistilBERT (distilbert-base-multilingual-cased)
- **Framework:** Hugging Face Transformers
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Experiment Tracking:** MLflow, DagsHub
- **Data Processing:** NLTK, spaCy, Pandas
- **Version Control:** Git, GitHub

## Project Structure
- `Config/` - configuration and constants
- `Data/` - raw and processed data
- `src/` - data ingestion, cleaning, preprocessing and model training
- `pipeline/` - training and prediction pipelines
- `main/` - FastAPI application
- `utils/` - model utility functions
- `streamlit_app.py` - Streamlit dashboard

## How to Run
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Start the API with `uvicorn main.app:app --reload`
5. Start the dashboard with `streamlit run streamlit_app.py`