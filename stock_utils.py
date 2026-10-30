"""
stock_utils.py  --  Modules 1, 2 and 3 of the project (version 2)
=================================================================
Module 1 : Data Collection              -> fetch_stock_data, handle_missing_values
Module 2 : Preprocessing + Features     -> add_indicators, build_feature_table
Module 3 : Model Building               -> RandomForest, GradientBoosting, LSTM (optional),
                                           Ensemble, evaluation, saving, prediction

Works for ANY Yahoo Finance symbol (stocks, indices, crypto, commodities, forex, ETFs).
Educational project only. NOT financial advice.
"""

import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
import ta                                   # technical analysis library
import joblib                               # saves .pkl files
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
STOCKS = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]    # default list used by train_models.py
MODEL_DIR = "models"
SEQ_LEN = 60                                     # LSTM looks at the last 60 days
TRAIN_RATIO = 0.80                               # 80% train, 20% test

# Features given to the models. Most are RATIOS, not raw prices.
# WHY: raw prices keep drifting upward over the years (non-stationary), so a model
# trained on old price levels does not transfer to new levels. Ratios such as
# "distance from MA20 in %" mean the same thing in any year. This is one of the
# main accuracy improvements over version 1.
FEATURES = [
    "Return", "Ret_lag1", "Ret_lag2", "Ret_lag5",     # today's and past daily returns
    "ROC10",                                          # 10-day rate of change
    "Vol10",                                          # 10-day volatility
    "Vol_ratio",                                      # today's volume / 20-day average volume
    "RSI", "Stoch",                                   # momentum oscillators
    "MACD_norm", "MACDhist_norm",                     # MACD scaled by price
    "Dist_MA20", "Dist_MA50", "MA20_vs_MA50",         # trend position
    "BB_width", "BB_pos",                             # Bollinger Band width and position
    "ATR_norm",                                       # average true range (volatility)
]


def safe_name(ticker):
    """'^NSEI' -> '_NSEI' so the name is safe for file names."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)


# ============================================================================
# MODULE 1 : DATA COLLECTION
# ============================================================================
def handle_missing_values(df):
    """
    Fill missing values WITHOUT looking into the future.
    VIVA: only forward-fill (copy yesterday to today). Back-fill would copy a
    FUTURE value into the past = data leakage.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()
    return df.dropna()


def fetch_stock_data(ticker, period="2y", start=None, end=None):
    """
    Download daily OHLCV data of ANY symbol using yfinance.
    Use either period ("1mo","3mo","6mo","1y","2y","5y","10y","max")
    or an explicit start/end date.
    """
    if start:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data received for '{ticker}'. Check the symbol and your internet connection.")

    # Newer yfinance returns 2-level column names like ("Close", "TCS.NS")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return handle_missing_values(df)


# ============================================================================
# MODULE 2 : PREPROCESSING + FEATURE ENGINEERING
# ============================================================================
def add_indicators(df):
    """
    Add technical indicators with the `ta` library. All use only PAST and
    PRESENT data (rolling windows look backwards) -> no leakage.
    """
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    # --- Price based (also used for charts) ---
    df["Return"] = close.pct_change()
    df["MA20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["MA50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # --- Extra features used by the models ---
    df["Ret_lag1"] = df["Return"].shift(1)          # return 1 day ago
    df["Ret_lag2"] = df["Return"].shift(2)
    df["Ret_lag5"] = df["Return"].shift(5)
    df["ROC10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    df["Vol10"] = df["Return"].rolling(10).std()
    vol_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)   # indices/forex may have 0 volume
    df["Stoch"] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
    df["MACD_norm"] = df["MACD"] / close
    df["MACDhist_norm"] = df["MACD_hist"] / close
    df["Dist_MA20"] = close / df["MA20"] - 1
    df["Dist_MA50"] = close / df["MA50"] - 1
    df["MA20_vs_MA50"] = df["MA20"] / df["MA50"] - 1
    band = (df["BB_high"] - df["BB_low"]).replace(0, np.nan)
    df["BB_width"] = band / df["MA20"]
    df["BB_pos"] = (close - df["BB_low"]) / band
    df["ATR_norm"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range() / close
    return df


def build_feature_table(df):
    """
    Indicators + Target. Target: Trend = 1 if Close(t+1) > Close(t) else 0.
    The LAST row has no tomorrow, so its Target is NaN (kept only for prediction).
    """
    df = add_indicators(df)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1.0, 0.0)
    df.iloc[-1, df.columns.get_loc("Target")] = np.nan
    return df.dropna(subset=FEATURES)


def time_split_index(n_rows, ratio=TRAIN_RATIO):
    """
    Chronological split: rows [0:split] = train (older), rows [split:] = test (newer).
    NO SHUFFLING. VIVA: shuffling would let the model train on days AFTER the
    test days, i.e. see the future (data leakage).
    """
    return int(n_rows * ratio)


def make_sequences(scaled_X, y, seq_len=SEQ_LEN):
    """Turn a 2-D table into 60-day windows for the LSTM: (samples, 60, features)."""
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
    MODEL A - RandomForest (BASELINE)
    VIVA: easy to explain (many trees voting), hard to overfit, works on small
    data, needs no scaling, gives feature importance (explainability). If a more
    complex model cannot beat it, the complexity is not worth it.
    Small depth + min_samples_leaf reduce overfitting.
    """
    return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=8,
                                  random_state=42, n_jobs=-1)


def build_gradient_boosting():
    """
    MODEL B - Gradient Boosting (HistGradientBoostingClassifier)
    Trees are built one after another; each new tree fixes the mistakes of the
    previous ones. Shallow trees + small learning rate + L2 regularisation keep it
    from memorising noise.
    """
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          l2_regularization=1.0, random_state=42)


def build_lstm(n_features, seq_len=SEQ_LEN):
    """
    MODEL C - LSTM (optional, needs TensorFlow)
    VIVA: stock data is a SEQUENCE. A RandomForest sees one row at a time; an LSTM
    reads 60 days in order and keeps a memory (cell state).
    Gates:  forget gate = what old memory to erase,
            input gate  = what new information to store,
            output gate = what part of memory to send out.
    New memory = old memory x forget + new info x input (mostly additive, so the
    gradient does not vanish like in a simple RNN).
    Architecture: Input(60 x features) -> LSTM(50) -> Dropout(0.2)
                  -> LSTM(50) -> Dropout(0.2) -> Dense(1, sigmoid), Adam optimizer.
    """
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
    """Return (metrics dictionary, confusion matrix as list)."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    return metrics, confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()


def walk_forward_accuracy(build_fn, X, y, n_splits=5):
    """
    Walk-forward validation: train on the past, test on the next block, repeat.
    Gives 5 accuracy numbers instead of 1, so we can see how stable the model is.
    """
    scores = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = build_fn()
        m.fit(X[tr], y[tr])
        scores.append(float(accuracy_score(y[te], m.predict(X[te]))))
    return scores


def train_lstm(X, y, split, scaler, epochs=40):
    """
    Train the LSTM. Returns (model, predictions, probabilities, y_test).
    Leakage care: scaler fitted on train rows only; a window is TRAIN if it ENDS
    before `split`; validation_split takes the LAST 10% (time order kept).
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    np.random.seed(42)
    tf.random.set_seed(42)

    scaled = scaler.transform(X)
    X_seq, y_seq, end_idx = make_sequences(scaled, y)
    is_train = end_idx < split
    model = build_lstm(X.shape[1])
    stop_early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(X_seq[is_train], y_seq[is_train], epochs=epochs, batch_size=32,
              validation_split=0.1, shuffle=False, callbacks=[stop_early], verbose=0)
    probs = model.predict(X_seq[~is_train], verbose=0).ravel()
    return model, (probs >= 0.5).astype(int), probs, y_seq[~is_train]


# ----------------------------------------------------------------------------
# Saving / loading
# ----------------------------------------------------------------------------
def _path(ticker, name):
    return os.path.join(MODEL_DIR, f"{safe_name(ticker)}_{name}")


def models_exist(ticker):
    return all(os.path.exists(_path(ticker, n)) for n in ("rf.pkl", "gb.pkl", "scaler.pkl", "results.pkl"))


def lstm_exists(ticker):
    return os.path.exists(_path(ticker, "lstm.h5"))


def load_results(ticker):
    p = _path(ticker, "results.pkl")
    return joblib.load(p) if os.path.exists(p) else None


def trained_tickers():
    """Symbols that already have saved results (used by the Model Performance page)."""
    if not os.path.isdir(MODEL_DIR):
        return []
    out = []
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.endswith("_results.pkl"):
            try:
                out.append(joblib.load(os.path.join(MODEL_DIR, f))["ticker"])
            except Exception:
                pass
    return out


def train_and_save(ticker, period="5y", epochs=40, use_lstm=True):
    """
    Whole pipeline for one symbol:
    fetch -> features -> time split -> train RF, GB (+LSTM) -> ensemble ->
    evaluate (test set + walk-forward) -> save.
    More history (period="10y" or "max") usually gives the models more to learn from.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    raw = fetch_stock_data(ticker, period)
    data = build_feature_table(raw).dropna(subset=["Target"])
    X = data[FEATURES].values
    y = data["Target"].values.astype(int)

    split = time_split_index(len(data))
    if split < SEQ_LEN + 30 or len(data) < 250:
        raise ValueError("Not enough history to train (need roughly 1 year or more of daily data). "
                         "Choose a longer period or a symbol with more history.")
    y_test = y[split:]
    probs = {}                                   # test-set P(UP) of every model
    models_out = {}

    # --- Model A: RandomForest ---
    rf = build_random_forest().fit(X[:split], y[:split])
    probs["rf"] = rf.predict_proba(X[split:])[:, list(rf.classes_).index(1)]
    joblib.dump(rf, _path(ticker, "rf.pkl"))
    importance = dict(zip(FEATURES, rf.feature_importances_.round(4).tolist()))

    # --- Model B: Gradient Boosting ---
    gb = build_gradient_boosting().fit(X[:split], y[:split])
    probs["gb"] = gb.predict_proba(X[split:])[:, list(gb.classes_).index(1)]
    joblib.dump(gb, _path(ticker, "gb.pkl"))

    # --- Scaler (train rows only) for the LSTM ---
    scaler = StandardScaler().fit(X[:split])
    joblib.dump(scaler, _path(ticker, "scaler.pkl"))

    # --- Model C: LSTM (optional) ---
    if use_lstm:
        try:
            model, _, lstm_probs, y_te = train_lstm(X, y, split, scaler, epochs)
            assert np.array_equal(y_te, y_test)
            model.save(_path(ticker, "lstm.h5"))
            probs["lstm"] = lstm_probs
        except ImportError:
            print("TensorFlow not installed -> LSTM skipped.")

    # --- Ensemble: average probability of all models ---
    probs["ensemble"] = np.mean([probs[k] for k in probs], axis=0)

    for key, p in probs.items():
        pred = (p >= 0.5).astype(int)
        m, cm = evaluate(y_test, pred)
        models_out[key] = {"pred": pred.tolist(), "prob": [round(float(v), 4) for v in p],
                           "metrics": m, "cm": cm}

    # --- Honest checks ---
    majority = max(y_test.mean(), 1 - y_test.mean())          # accuracy of "always guess the commoner class"
    wf = {"RandomForest": walk_forward_accuracy(build_random_forest, X, y),
          "GradientBoosting": walk_forward_accuracy(build_gradient_boosting, X, y)}

    results = {
        "ticker": ticker, "period": period,
        "n_train": int(split), "n_test": int(len(y_test)),
        "start_date": data.index[0].strftime("%Y-%m-%d"),
        "end_date": data.index[-1].strftime("%Y-%m-%d"),
        "dates": [d.strftime("%Y-%m-%d") for d in data.index[split:]],
        "y_test": y_test.tolist(),
        "baseline_accuracy": round(float(majority), 4),
        "walk_forward": wf,
        "importance": importance,
        "models": models_out,
        # keys kept for the notebook / older code
        "rf": models_out["rf"], "lstm": models_out.get("lstm"),
    }
    joblib.dump(results, _path(ticker, "results.pkl"))
    return results


# ----------------------------------------------------------------------------
# Prediction for tomorrow (next trading day)
# ----------------------------------------------------------------------------
def to_trend(prob_up):
    """P(UP) -> ("UP"/"DOWN", confidence %)."""
    if prob_up >= 0.5:
        return "UP", int(round(prob_up * 100))
    return "DOWN", int(round((1 - prob_up) * 100))


def confidence_label(conf):
    """Plain-words reading of the confidence number (honesty for beginners)."""
    if conf < 55:
        return "Very low conviction - almost a coin flip"
    if conf < 60:
        return "Low conviction"
    if conf < 70:
        return "Moderate conviction"
    return "Higher conviction (still not a guarantee)"


def predict_tomorrow(ticker):
    """Load saved models + latest data and return P(UP) from each model."""
    table = build_feature_table(fetch_stock_data(ticker, "2y"))
    X_all = table[FEATURES].values

    rf = joblib.load(_path(ticker, "rf.pkl"))
    gb = joblib.load(_path(ticker, "gb.pkl"))
    out = {
        "last_date": table.index[-1].strftime("%d %b %Y"),
        "last_close": float(table["Close"].iloc[-1]),
        "rf_prob_up": float(rf.predict_proba(X_all[-1:])[0][list(rf.classes_).index(1)]),
        "gb_prob_up": float(gb.predict_proba(X_all[-1:])[0][list(gb.classes_).index(1)]),
        "lstm_prob_up": None,
    }
    if lstm_exists(ticker):
        try:
            from tensorflow.keras.models import load_model
            scaler = joblib.load(_path(ticker, "scaler.pkl"))
            window = scaler.transform(X_all[-SEQ_LEN:])[np.newaxis, :, :]
            lstm = load_model(_path(ticker, "lstm.h5"), compile=False)
            out["lstm_prob_up"] = float(lstm.predict(window, verbose=0)[0][0])
        except ImportError:
            pass
    parts = [out["rf_prob_up"], out["gb_prob_up"]] + ([out["lstm_prob_up"]] if out["lstm_prob_up"] is not None else [])
    out["ens_prob_up"] = float(np.mean(parts))
    return out
