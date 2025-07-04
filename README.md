# ✈️ Flight Fare Prediction

A machine learning project that predicts airline ticket prices based on route, airline, duration, and other features.

## 🔍 Overview
This project simulates a pricing prototype for the airline industry. It uses a Random Forest regression model trained on historical flight data.

## 📊 Tech Stack
- Python, Pandas, scikit-learn
- Streamlit (for UI)

## 🧠 Features
- Predict fare for any airline-route combination
- Cleaned + encoded features
- Deployed as a user-facing Streamlit app

## 📁 Data
Dataset used: [Flight Price Prediction - Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)  
> Note: Due to license restrictions, please download manually.

## 📦 Run it locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
