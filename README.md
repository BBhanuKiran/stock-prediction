# Intelligent Stock Market Trend Predictor - All-in-One Learning & Analysis Website

BTech CSE / AIML Final Year Project by **Bhanu Kiran Bonam** - **Educational purpose only. Not financial advice.**

A premium-looking Streamlit website where beginners can **learn trading from scratch**, check **any market**
(Indian stocks, indices, US stocks, crypto, commodities, forex, ETFs), view **technical charts**, get an
**ML next-day trend estimate** (UP/DOWN) and learn **how not to lose money**.

## Pages

| Page | What it gives you |
|---|---|
| Home | Market snapshot (NIFTY, SENSEX, S&P 500, Bitcoin, Gold, USD/INR) and where-to-start guide |
| Market Explorer | Category + search + duration (1M to Max or custom dates), stats, risk level, company info, news, compare, CSV download |
| Technical Analysis | Candlestick/line, MA20/MA50, Bollinger, volume, RSI, MACD, drawdown, signal summary with cautions - for any symbol |
| Predict Trend | Train on 2y/5y/10y/max history, RandomForest + Gradient Boosting (+ LSTM optional) + Ensemble, honest reality check |
| Model Performance | Metrics, coin-flip and baseline comparison, walk-forward validation, confusion matrices, actual vs predicted |
| Learn Trading | 11-module course, 15-question quiz, glossary, free resources |
| Broker & App Guide | Groww, Zerodha, Angel One, Upstox, Dhan and more (neutral), account opening steps, how to choose |
| Safety Center | 13 golden rules, scam detector, position-size / risk-reward / SIP / growth calculators, loss-recovery table |
| Team Details | Developer details |

## Files

| File | Purpose |
|---|---|
| `app.py` | Module 5 - the website |
| `stock_utils.py` | Modules 1-3 - data collection, feature engineering, models |
| `market_data.py` | Asset catalog, search, company info, news, risk metrics, signal summary |
| `visualization.py` | Module 4 - Plotly / Matplotlib charts |
| `learn_content.py` | Course, quiz, glossary, broker guide, safety content |
| `train_models.py` | Optional command-line pre-training |
| `model_training.ipynb` | Step-by-step notebook for viva explanation |
| `.streamlit/config.toml` | Navy and gold theme |
| `requirements.txt` | Libraries |

## Run on your computer (Windows PowerShell)

```
python --version                      (use 3.10, 3.11 or 3.12)
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Open the site, go to **Predict Trend**, choose any asset and click **Train / retrain models**.
(Optional: `python train_models.py` trains a few assets in advance.)

If TensorFlow fails to install, delete the `tensorflow` line from `requirements.txt`. The site still works
(RandomForest + Gradient Boosting + Ensemble); only the LSTM is skipped.

## Deploy free (Streamlit Community Cloud)
1. Remove the `tensorflow` and `jupyter` lines from `requirements.txt`.
2. Push the folder to a public GitHub repository (`app.py` and `requirements.txt` at the top level; include the `.streamlit` folder).
3. On share.streamlit.io choose the repo, main file `app.py`, Python 3.11/3.12, then Deploy.
Models are re-trained from the Predict Trend page (takes seconds without LSTM); cloud storage resets when the app restarts.

## How accuracy was improved (viva points)
1. **More history:** train on 5y / 10y / all available data instead of 2 years.
2. **Stationary features:** ratios (distance from MA, volatility, volume ratio, Bollinger position) instead of raw prices.
3. **More models:** RandomForest, Gradient Boosting, optional LSTM and an Ensemble (average probability).
4. **Honest evaluation:** time-based 80:20 split, walk-forward validation, and comparison with a coin flip and an "always guess the commoner side" baseline.

Next-day direction is close to random. 50-58% accuracy is a realistic, honest result. 100% accuracy is impossible, and no tool can guarantee profit.

## Data leakage prevention
No shuffling; time-based split; scaler fitted on train rows only; forward-fill only; indicators use only past data; last row (unknown tomorrow) excluded from training.

## Future scope
News sentiment analysis (VADER / FinBERT), more indicators, hyper-parameter tuning, Transformer models, portfolio tracker, alerts.

**Disclaimer:** Educational use only. Not financial advice. You can lose money in markets. Predictions come from past prices and can be wrong.
