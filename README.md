# Intelligent Stock Market Trend Predictor using ML

BTech CSE / AIML Final Year Project - **Educational purpose only. Not financial advice.**

Predicts the **next-day trend (UP / DOWN)** of NSE stocks (TCS, RELIANCE, INFY) using
technical indicators, a **RandomForest** baseline and an **LSTM** model, and shows
everything in a **Streamlit** web app.

## Project structure

| File | Purpose |
|---|---|
| `stock_utils.py` | Module 1 (data collection), Module 2 (features), Module 3 (models) |
| `visualization.py` | Module 4 - Plotly and Matplotlib charts |
| `app.py` | Module 5 - Streamlit web app |
| `train_models.py` | Trains and saves models for all 3 stocks |
| `model_training.ipynb` | Step-by-step notebook for viva explanation |
| `requirements.txt` | Libraries to install |
| `models/` | Created automatically: `.pkl` (RandomForest, scaler, results) and `.h5` (LSTM) |

## How to run

1. Install **Python 3.10 or 3.11** (TensorFlow does not support very new Python versions).
2. Open a terminal inside this folder and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Mac/Linux)
   ```
3. Install libraries:
   ```
   pip install -r requirements.txt
   ```
4. (Optional but recommended) Understand the steps in the notebook:
   ```
   jupyter notebook model_training.ipynb
   ```
5. Train and save the models (needs internet, takes a few minutes):
   ```
   python train_models.py
   ```
6. Start the web app:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501. You can also train a single stock from the
   **Predict Trend** tab using the "Train / Retrain" button.

## Common problems

- **"No data received"**: check internet, or upgrade yfinance: `pip install -U yfinance`.
- **TensorFlow install fails**: use Python 3.10/3.11. Without TensorFlow the app still works with RandomForest only.
- **Accuracy is only about 50-55%**: this is normal. Next-day stock direction is very hard to predict and the test set is small (about 90 days). Report your real numbers honestly in the viva.
- Edit the team and guide details in `app.py` (search for `TODO`).

## Key viva concepts (explained in code comments)

- Why RandomForest is the baseline -> `build_random_forest()` in `stock_utils.py`
- Why LSTM suits time series, and the forget/input/output gates -> `build_lstm()`
- How data leakage is prevented -> `time_split_index()`, `handle_missing_values()`, `train_lstm()`

## Future scope

- Sentiment analysis from news headlines as an extra feature
- More stocks and the NIFTY 50 index, longer history
- Walk-forward validation, hyperparameter tuning, GRU / Transformer models

**Disclaimer:** Predictions come from models trained on historical prices and can be wrong. Do not use them for investment decisions.
