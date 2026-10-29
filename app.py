import sys

import streamlit as st

# ----------------------------------------------------------------------------
# Page setup + constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Intelligent Stock Market Trend Predictor",
                   page_icon="📈", layout="wide")

try:
    import stock_utils as su
    import visualization as viz
except (ImportError, OSError) as e:
    if "Application Control policy has blocked" in str(e):
        st.error(
            "Windows Application Control blocked a required package file "
            "(pandas). The app cannot run until your administrator approves "
            "this Python environment or provides an approved environment. "
            "After approval, install the project dependencies and restart the app."
        )
    else:
        st.error(
            "A required Python package could not be loaded. Activate the project's "
            "supported Python 3.10 or 3.11 environment, install dependencies with "
            "`python -m pip install -r requirements.txt`, then restart the app."
        )
    st.code(f"{type(e).__name__}: {e}")
    st.stop()
    print(
        "Could not start the app. Run it with "
        "`python -m streamlit run app.py`. "
        f"Underlying error: {type(e).__name__}: {e}",
        file=sys.stderr,
    )
    raise SystemExit(1)

DISCLAIMER = ("⚠️ **Disclaimer:** This project is for **educational purposes only** and "
              "is **not financial advice**. Predictions come from machine learning models "
              "trained on past prices and can be wrong. Do not make investment decisions "
              "based on this app.")

# TODO: edit these details before your presentation
TEAM = [
    {"Name": "Bhanu Kiran Bonam", "Roll No": "Enter roll number", "Role": "Team member"},
    {"Name": "Enter name", "Roll No": "Enter roll number", "Role": "Team member"},
]
GUIDE = "Enter guide name"
COLLEGE = "DNR College of Engineering and Technology, Bhimavaram"
DEPARTMENT = "B.Tech Computer Science and Engineering (AIML), 2022-2026"


@st.cache_data(ttl=300, show_spinner="Fetching data from Yahoo Finance...")
def load_data(ticker, period):
    """Download data once and reuse it for 5 minutes (makes the app faster)."""
    return su.fetch_stock_data(ticker, period)


# ----------------------------------------------------------------------------
# Sidebar: stock selector + period selector
# ----------------------------------------------------------------------------
st.sidebar.title("📈 Settings")
ticker = st.sidebar.selectbox("Select stock", su.STOCKS)
period = st.sidebar.selectbox("Select period", ["6mo", "1y", "2y", "5y"], index=2)
st.sidebar.markdown("---")
st.sidebar.caption("Educational project. Not financial advice.")

st.title("Intelligent Stock Market Trend Predictor using ML")

try:
    raw = load_data(ticker, period)
except Exception as e:                      # no internet, wrong ticker, etc.
    st.error(f"Could not load data: {e}")
    st.stop()

if len(raw) < 2:
    st.error("Not enough historical data was returned to display live market changes.")
    st.stop()

data = su.add_indicators(raw)               # raw prices + indicators (for charts)

tab_home, tab_live, tab_tech, tab_pred, tab_perf, tab_team = st.tabs(
    ["Home", "Live Data", "Technical Charts", "Predict Trend",
     "Model Performance", "Team Details"])

# ----------------------------------------------------------------------------
# TAB 1 : Home
# ----------------------------------------------------------------------------
with tab_home:
    st.header("Welcome")
    st.write(
        "This app predicts the **next-day trend (UP or DOWN)** of NSE stocks "
        "(TCS, Reliance, Infosys) using technical indicators and two machine "
        "learning models: **RandomForest** (baseline) and **LSTM** (deep learning).")
    st.subheader("How it works")
    st.markdown(
        "1. **Data Collection** - daily prices from Yahoo Finance (yfinance)\n"
        "2. **Feature Engineering** - MA20, MA50, RSI, MACD, Bollinger Bands, lag prices\n"
        "3. **Model Training** - first 80% of days train, last 20% test (no shuffling)\n"
        "4. **Evaluation** - Accuracy, Precision, Recall, F1, Confusion Matrix\n"
        "5. **Prediction** - tomorrow's trend with a confidence percentage")
    st.info("Use the sidebar to choose a stock, then open the other tabs.")
    st.warning(DISCLAIMER)

# ----------------------------------------------------------------------------
# TAB 2 : Live Data
# ----------------------------------------------------------------------------
with tab_live:
    st.header(f"{ticker} - Latest Market Data")
    st.caption("Data from Yahoo Finance, usually delayed by about 15 minutes.")
    last, prev = raw.iloc[-1], raw.iloc[-2]
    change = last["Close"] - prev["Close"]
    change_pct = change / prev["Close"] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close (INR)", f"{last['Close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    c2.metric("Day High", f"{last['High']:.2f}")
    c3.metric("Day Low", f"{last['Low']:.2f}")
    c4.metric("Volume", f"{int(last['Volume']):,}")
    st.line_chart(raw["Close"])
    st.subheader("Recent data")
    st.dataframe(raw.tail(15).round(2))

# ----------------------------------------------------------------------------
# TAB 3 : Technical Charts
# ----------------------------------------------------------------------------
with tab_tech:
    st.header(f"{ticker} - Technical Charts")
    st.plotly_chart(viz.candlestick_chart(data, ticker))
    st.plotly_chart(viz.rsi_chart(data, ticker))
    st.plotly_chart(viz.macd_chart(data, ticker))
    st.caption("RSI above 70 = overbought, below 30 = oversold. "
               "MACD crossing above its signal line is usually read as bullish.")

# ----------------------------------------------------------------------------
# TAB 4 : Predict Trend
# ----------------------------------------------------------------------------
with tab_pred:
    st.header("Predict Tomorrow's Trend")

    if not su.models_exist(ticker):
        st.warning(f"No trained model found for {ticker}. Click the button below "
                   "to train (takes about 1-2 minutes).")
    if st.button(f"Train / Retrain models for {ticker}"):
        with st.spinner("Training RandomForest and LSTM... please wait"):
            try:
                su.train_and_save(ticker, "2y")
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if su.models_exist(ticker):
        choice = st.radio("Model for the headline prediction",
                          ["RandomForest", "LSTM"], horizontal=True)
        if st.button("Predict tomorrow's trend"):
            try:
                with st.spinner("Loading latest data and models..."):
                    out = su.predict_tomorrow(ticker)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            probs = {"RandomForest": out["rf_prob_up"], "LSTM": out["lstm_prob_up"]}
            if probs[choice] is None:
                st.warning("LSTM is not available (TensorFlow missing or not trained). "
                           "Showing RandomForest instead.")
                choice = "RandomForest"

            trend, conf = su.to_trend(probs[choice])
            message = f"{ticker} - Tomorrow: {trend} Trend (Confidence {conf}%)"
            if trend == "UP":
                st.success(f"### {message}")
            else:
                st.error(f"### {message}")
            st.caption(f"Model: {choice} | Based on data up to {out['last_date']} "
                       f"(last close {out['last_close']:.2f} INR)")

            col1, col2 = st.columns(2)
            for col, name in zip((col1, col2), ("RandomForest", "LSTM")):
                p = probs[name]
                if p is None:
                    col.metric(name, "Not available")
                else:
                    t, c = su.to_trend(p)
                    col.metric(name, f"{t} ({c}%)", f"P(UP) = {p:.2f}", delta_color="off")

        res = su.load_results(ticker)
        if res:
            st.subheader("Why? Indicators the RandomForest relied on")
            st.plotly_chart(viz.feature_importance_chart(res["importance"]))
    st.warning(DISCLAIMER)

# ----------------------------------------------------------------------------
# TAB 5 : Model Performance
# ----------------------------------------------------------------------------
with tab_perf:
    st.header(f"Model Performance - {ticker}")
    res = su.load_results(ticker)
    if res is None:
        st.info("Train the models first (Predict Trend tab, or run `python train_models.py`).")
    else:
        st.write(f"Training days: **{res['n_train']}** | "
                 f"Test days (most recent 20%): **{res['n_test']}**")

        models = {"RandomForest": res["rf"]}
        if res["lstm"]:
            models["LSTM"] = res["lstm"]

        table = [
            {"Model": name, **model["metrics"]}
            for name, model in models.items()
        ]
        st.dataframe(table)
        for name, m in models.items():
            acc = m["metrics"]["Accuracy"] * 100
            st.write(f"- {name}: accuracy **{acc:.1f}%** "
                     f"({'meets' if acc > 55 else 'below'} the 55% objective)")
        st.caption("The test set is small, so accuracy changes a lot from run to run. "
                   "Around 50-55% is normal for next-day stock direction.")

        st.plotly_chart(viz.model_comparison_bar({n: m["metrics"] for n, m in models.items()}))

        cols = st.columns(len(models))
        for col, (name, m) in zip(cols, models.items()):
            with col:
                st.pyplot(viz.confusion_matrix_figure(m["cm"], name))

        st.plotly_chart(viz.actual_vs_predicted_chart(
            res["dates"], res["y_test"], {n: m["pred"] for n, m in models.items()}))

# ----------------------------------------------------------------------------
# TAB 6 : Team Details
# ----------------------------------------------------------------------------
with tab_team:
    st.header("Team Details")
    st.write(f"**Project:** Intelligent Stock Market Trend Predictor using ML")
    st.write(f"**College:** {COLLEGE}")
    st.write(f"**Department:** {DEPARTMENT}")
    st.write(f"**Guide:** {GUIDE}")
    st.table(TEAM)
    st.warning(DISCLAIMER)
