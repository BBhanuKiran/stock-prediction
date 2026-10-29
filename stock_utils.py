"""
stock_utils.py  --  Modules 1, 2 and 3 of the project
======================================================
Module 1 : Data Collection              -> fetch_stock_data, handle_missing_values
Module 2 : Preprocessing + Features     -> add_indicators, build_feature_table
Module 3 : Model Building               -> RandomForest (Model A) + LSTM (Model B),
                                           evaluation, saving, loading, prediction

Educational project only. NOT financial advice.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import ta                                   # technical analysis library
import joblib                               # used to save .pkl files
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ----------------------------------------------------------------------------
# SETTINGS (change here, everything else follows)
# ----------------------------------------------------------------------------
STOCKS = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]   # NSE tickers end with ".NS"
MODEL_DIR = "models"                             # folder where models are saved
SEQ_LEN = 60                                     # LSTM looks at the last 60 days
TRAIN_RATIO = 0.80                               # 80% train, 20% test

# Columns given to the models as input (X). "Target" is the answer (y).
FEATURES = [
    "Close", "Volume", "Return",
    "MA20", "MA50", "RSI",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_high", "BB_low",
    "Close_lag1", "Close_lag2", "Close_lag5",
]


# ============================================================================
# MODULE 1 : DATA COLLECTION
# ============================================================================
def handle_missing_values(df):
    """
    Fill missing values WITHOUT looking into the future.

    VIVA POINT (data leakage): we use forward-fill only (copy yesterday's value
    to today). We do NOT use back-fill, because back-fill would copy a FUTURE
    value into the past, which leaks future information into training.
    """
    df = df.replace([np.inf, -np.inf], np.nan)   # infinity -> missing
    df = df.ffill()                              # fill gaps with previous day
    df = df.dropna()                             # drop rows still empty (very first rows)
    return df


def fetch_stock_data(ticker, period="2y"):
    """Download daily OHLCV data of one NSE stock using yfinance."""
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data received for {ticker}. "
                         "Check your internet connection or the ticker symbol.")

    # Newer yfinance versions return 2-level column names, e.g. ("Close", "TCS.NS").
    # We keep only the first level so that df["Close"] works normally.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)   # remove time zone
    df = df[~df.index.duplicated(keep="first")].sort_index()  # no duplicate dates
    return handle_missing_values(df)


# ============================================================================
# MODULE 2 : PREPROCESSING + FEATURE ENGINEERING
# ============================================================================
def add_indicators(df):
    """
    Add technical indicators using the `ta` library.
    Every indicator below uses only PAST and PRESENT prices (rolling windows
    look backwards), so there is no future information in them.
    """
    df = df.copy()
    close = df["Close"]

    # Daily percentage change
    df["Return"] = close.pct_change()

    # Moving averages: average closing price of the last 20 / 50 days
    df["MA20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["MA50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    # RSI(14): momentum between 0 and 100 (>70 overbought, <30 oversold)
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # MACD: difference of 12-day and 26-day exponential averages
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # Bollinger Bands: 20-day average +/- 2 standard deviations
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # Lag features: the closing price 1, 2 and 5 days ago
    df["Close_lag1"] = close.shift(1)
    df["Close_lag2"] = close.shift(2)
    df["Close_lag5"] = close.shift(5)
    return df


def build_feature_table(df):
    """
    Indicators + Target, with rows that have missing indicator values removed.

    Target (what we want to predict):
        Trend = 1 if Close(t+1) > Close(t) else 0
        i.e. 1 = tomorrow's close is higher than today's (UP), 0 = DOWN.

    The LAST row has no tomorrow yet, so its Target is NaN. We keep that row
    because it is the one we use to predict tomorrow in the web app.
    """
    df = add_indicators(df)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1.0, 0.0)
    df.iloc[-1, df.columns.get_loc("Target")] = np.nan      # tomorrow unknown
    df = df.dropna(subset=FEATURES)                          # first ~50 rows have no MA50
    return df


def time_split_index(n_rows, ratio=TRAIN_RATIO):
    """
    Chronological split point. Rows [0:split] = train (older days),
    rows [split:] = test (newer days). NO SHUFFLING.

    VIVA POINT (data leakage): if we shuffled, a model could train on
    Friday's data and be tested on Thursday's data, i.e. it would "see the
    future". In real life we only know the past, so the test set must be the
    most recent part of the timeline.
    """
    return int(n_rows * ratio)


def make_sequences(scaled_X, y, seq_len=SEQ_LEN):
    """
    Convert a 2-D table into 3-D windows for the LSTM.
    Window t = rows [t-59 ... t] (60 days) and the label is y[t].
    Shape of result: (samples, 60 days, number of features).
    Also returns end_idx = the row number where each window ends, which helps
    us split windows into train/test later without leakage.
    """
    X_seq, y_seq, end_idx = [], [], []
    for t in range(seq_len - 1, len(scaled_X)):
        X_seq.append(scaled_X[t - seq_len + 1: t + 1])
        y_seq.append(y[t])
        end_idx.append(t)
    return np.array(X_seq), np.array(y_seq), np.array(end_idx)


# ============================================================================
# MODULE 3 : MODEL BUILDING
# ============================================================================
def build_random_forest():
    """
    MODEL A - RandomForestClassifier (BASELINE)

    VIVA: Why RandomForest as the baseline?
      1. Easy to explain: it is many decision trees voting together.
      2. Hard to overfit compared to one tree, and works well on small data
         (we only have ~500 rows).
      3. Needs no feature scaling and gives feature_importances_, so we can
         say WHICH indicator mattered. This supports "explainable" trends.
      4. A good baseline: if a complex LSTM cannot beat it, the LSTM is not
         worth the extra complexity.
    max_depth and min_samples_leaf are kept small on purpose to reduce overfitting.
    """
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=5, random_state=42)


def build_lstm(n_features, seq_len=SEQ_LEN):
    """
    MODEL B - LSTM (Long Short-Term Memory)

    VIVA: Why LSTM for time series?
      Stock data is a SEQUENCE: today's movement depends on the order of the
      previous days. A RandomForest sees one row at a time and has no memory.
      An LSTM reads 60 days in order and keeps a memory (cell state).

    VIVA: Explain the 3 gates of an LSTM cell
      * Forget gate : looks at yesterday's output + today's input and decides
                      what part of the old memory to ERASE.
      * Input gate  : decides what NEW information from today to STORE.
      * Output gate : decides what part of the memory to SEND OUT as today's
                      output (hidden state).
      Memory update : new memory = (old memory x forget) + (new info x input).
      Because memory is mostly "added", gradients do not vanish easily; this is
      why LSTM beats a simple RNN on long sequences.

    Architecture (as per project requirement):
      Input (60 days x features) -> LSTM(50) -> Dropout(0.2)
                                 -> LSTM(50) -> Dropout(0.2) -> Dense(1, sigmoid)
      * return_sequences=True on the 1st LSTM passes all 60 outputs to the 2nd.
      * Dropout(0.2) randomly switches off 20% neurons while training
        -> reduces overfitting.
      * Dense(1) + sigmoid gives a probability between 0 and 1 (1 = UP).
      * Adam optimizer + binary_crossentropy loss, because it is a 2-class problem.
    """
    # TensorFlow is imported here (not at the top) so that the rest of the
    # project still works even if TensorFlow is not installed.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate(y_true, y_pred):
    """Return (metrics dictionary, confusion matrix as a list)."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics, cm


def train_lstm(X, y, split, scaler, epochs=50):
    """
    Train the LSTM and return (model, predictions, probabilities, y_test).

    DATA LEAKAGE CARE:
      * `scaler` was fitted on the TRAIN rows only (done by the caller).
      * A window belongs to TRAIN if it ENDS before `split`, otherwise to TEST.
        A test window may contain older days from the training period as
        context; that is fine because those days are in the past.
      * validation_split takes the LAST 10% of training windows (still in time
        order) and shuffle=False keeps the time order.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    np.random.seed(42)
    tf.random.set_seed(42)

    scaled = scaler.transform(X)
    X_seq, y_seq, end_idx = make_sequences(scaled, y)
    is_train = end_idx < split
    X_tr, y_tr = X_seq[is_train], y_seq[is_train]
    X_te, y_te = X_seq[~is_train], y_seq[~is_train]

    model = build_lstm(X.shape[1])
    stop_early = EarlyStopping(monitor="val_loss", patience=8,
                               restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=16,
              validation_split=0.1, shuffle=False,
              callbacks=[stop_early], verbose=0)

    probs = model.predict(X_te, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    return model, pred, probs, y_te


# ----------------------------------------------------------------------------
# Saving / loading helpers
# ----------------------------------------------------------------------------
def _path(ticker, name):
    return os.path.join(MODEL_DIR, f"{ticker}_{name}")


def models_exist(ticker):
    """True if at least the RandomForest model and scaler are saved."""
    return os.path.exists(_path(ticker, "rf.pkl")) and os.path.exists(_path(ticker, "scaler.pkl"))


def lstm_exists(ticker):
    return os.path.exists(_path(ticker, "lstm.h5"))


def load_results(ticker):
    """Load the saved evaluation results (metrics, confusion matrices ...)."""
    p = _path(ticker, "results.pkl")
    return joblib.load(p) if os.path.exists(p) else None


def train_and_save(ticker, period="2y", epochs=50):
    """
    Run the WHOLE pipeline for one stock:
    fetch -> features -> split -> train RF -> train LSTM -> evaluate -> save.
    Saves  models/<ticker>_rf.pkl, _scaler.pkl, _lstm.h5, _results.pkl
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Steps 1-2: data + features
    raw = fetch_stock_data(ticker, period)
    table = build_feature_table(raw)
    data = table.dropna(subset=["Target"])          # drop last row (tomorrow unknown)
    X = data[FEATURES].values
    y = data["Target"].values.astype(int)

    # Step 3: time-based split (no shuffle)
    split = time_split_index(len(data))
    if split < SEQ_LEN + 30:
        raise ValueError("Not enough data to train. Use a longer period (e.g. 2y).")
    y_test = y[split:]
    test_dates = [d.strftime("%Y-%m-%d") for d in data.index[split:]]

    # Step 4a: Model A - RandomForest
    rf = build_random_forest()
    rf.fit(X[:split], y[:split])
    rf_pred = rf.predict(X[split:])
    rf_metrics, rf_cm = evaluate(y_test, rf_pred)
    joblib.dump(rf, _path(ticker, "rf.pkl"))
    importance = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

    # Scaler is fitted on TRAIN rows only (prevents leakage), used by the LSTM
    scaler = StandardScaler().fit(X[:split])
    joblib.dump(scaler, _path(ticker, "scaler.pkl"))

    # Step 4b: Model B - LSTM (skipped politely if TensorFlow is missing)
    lstm_res = None
    try:
        model, lstm_pred, _, y_te = train_lstm(X, y, split, scaler, epochs)
        assert np.array_equal(y_te, y_test), "RF and LSTM test sets must match"
        model.save(_path(ticker, "lstm.h5"))
        m, cm = evaluate(y_test, lstm_pred)
        lstm_res = {"pred": lstm_pred.tolist(), "metrics": m, "cm": cm}
    except ImportError:
        print("TensorFlow not installed -> LSTM skipped (RandomForest still works).")

    # Step 5: evaluation results saved for the web app
    results = {
        "ticker": ticker,
        "n_train": int(split),
        "n_test": int(len(y_test)),
        "dates": test_dates,
        "y_test": y_test.tolist(),
        "rf": {"pred": rf_pred.tolist(), "metrics": rf_metrics, "cm": rf_cm},
        "lstm": lstm_res,
        "importance": importance,
    }
    joblib.dump(results, _path(ticker, "results.pkl"))
    return results


# ----------------------------------------------------------------------------
# Prediction for tomorrow
# ----------------------------------------------------------------------------
def to_trend(prob_up):
    """Probability of UP -> ("UP"/"DOWN", confidence in %)."""
    if prob_up >= 0.5:
        return "UP", int(round(prob_up * 100))
    return "DOWN", int(round((1 - prob_up) * 100))


def predict_tomorrow(ticker):
    """
    Use saved models + the latest market data to predict tomorrow's trend.
    Returns a dictionary with last date, last close and P(UP) of each model.
    """
    raw = fetch_stock_data(ticker, "1y")           # 1 year is enough for MA50 + 60 days
    table = build_feature_table(raw)
    latest_X = table[FEATURES].values              # all rows with valid features

    # RandomForest uses only the latest day (one row)
    rf = joblib.load(_path(ticker, "rf.pkl"))
    up_col = list(rf.classes_).index(1)
    rf_prob = float(rf.predict_proba(latest_X[-1:])[0][up_col])

    # LSTM uses the latest 60 days (scaled with the TRAIN scaler)
    lstm_prob = None
    if lstm_exists(ticker):
        try:
            from tensorflow.keras.models import load_model
            scaler = joblib.load(_path(ticker, "scaler.pkl"))
            window = scaler.transform(latest_X[-SEQ_LEN:])[np.newaxis, :, :]
            lstm = load_model(_path(ticker, "lstm.h5"), compile=False)
            lstm_prob = float(lstm.predict(window, verbose=0)[0][0])
        except ImportError:
            lstm_prob = None

    return {
        "last_date": table.index[-1].strftime("%d %b %Y"),
        "last_close": float(table["Close"].iloc[-1]),
        "rf_prob_up": rf_prob,
        "lstm_prob_up": lstm_prob,
    }
