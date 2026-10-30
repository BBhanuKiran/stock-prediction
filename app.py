"""
app.py  --  Module 5 : Streamlit Web Application (Intelligent Stock Market Trend Predictor)
===========================================================================================
Run with:   streamlit run app.py

Pages: Home | Market Explorer | Technical Analysis | Predict Trend | Model Performance |
       Learn Trading | Broker & App Guide | Safety Center | Team Details

Educational project only. NOT financial advice.
"""

import datetime as dt
import importlib.util
import math

import numpy as np
import pandas as pd
import streamlit as st

import learn_content as lc
import market_data as md
import stock_utils as su
import visualization as viz

# ----------------------------------------------------------------------------
# Page setup + premium styling (navy + gold, classic serif headings)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Intelligent Stock Market Trend Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; letter-spacing: 0.2px;}
h1 {color: #C9A227;}
.hero {background: linear-gradient(135deg, #14213D 0%, #0B1220 70%); border: 1px solid #2A3550;
       border-left: 5px solid #C9A227; border-radius: 12px; padding: 28px 32px; margin-bottom: 18px;}
.hero h1 {margin: 0 0 6px 0; font-size: 2.1rem;}
.hero p {color: #B8C1D6; margin: 0; font-size: 1.05rem;}
.pill {display: inline-block; padding: 2px 12px; border-radius: 20px; font-size: 0.8rem;
       border: 1px solid #C9A227; color: #C9A227; margin-right: 6px;}
div[data-testid="stMetric"] {background: #131C2E; border: 1px solid #25304A; border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] {color: #9AA6BF;}
section[data-testid="stSidebar"] {border-right: 1px solid #25304A;}
.small {color: #8A94A6; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

PAGES = ["Home", "Market Explorer", "Technical Analysis", "Predict Trend", "Model Performance",
         "Learn Trading", "Broker & App Guide", "Safety Center", "Team Details"]

DURATIONS = {  # label: (yfinance period, days to show, longer period fetched so indicators are accurate)
    "1 Month": ("1mo", 31, "1y"), "3 Months": ("3mo", 92, "1y"), "6 Months": ("6mo", 183, "2y"),
    "1 Year": ("1y", 365, "2y"), "2 Years": ("2y", 730, "5y"), "5 Years": ("5y", 1826, "10y"),
    "10 Years": ("10y", 3652, "max"), "Max (all data)": ("max", None, "max"),
}
MODEL_NAMES = {"rf": "RandomForest", "gb": "GradientBoosting", "lstm": "LSTM", "ensemble": "Ensemble"}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def go_to(page):
    """Button callback: switch page."""
    st.session_state["page"] = page


def money(value, symbol):
    return f"{symbol}{value:,.2f}"


def footer():
    st.divider()
    st.caption("Educational use only. Not financial advice. You can lose money in markets.")


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch(ticker, period, start, end):
    return su.fetch_stock_data(ticker, period=period, start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query):
    return md.search_online(query)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_info(ticker):
    return md.get_info(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker):
    return md.get_news(ticker)


@st.cache_data(ttl=300, show_spinner="Loading market snapshot...")
def snapshot():
    rows = []
    for name, sym in md.SNAPSHOT.items():
        try:
            close = su.fetch_stock_data(sym, "1mo")["Close"]
            rows.append((name, float(close.iloc[-1]), float((close.iloc[-1] / close.iloc[-2] - 1) * 100)))
        except Exception:
            pass
    return rows


def get_prices(ticker, dur):
    """
    Download prices and add indicators. We fetch EXTRA history before the chosen
    window so that MA50, RSI, MACD are already 'warmed up' at the start of the chart.
    """
    if dur["label"] == "Custom range":
        start, end = pd.Timestamp(dur["start"]), pd.Timestamp(dur["end"]) + pd.Timedelta(days=1)
        if start >= end:
            raise ValueError("'From' date must be before 'To' date.")
        raw = cached_fetch(ticker, "2y", (start - pd.Timedelta(days=250)).strftime("%Y-%m-%d"),
                           end.strftime("%Y-%m-%d"))
        cutoff = start
    else:
        _, days, warm = DURATIONS[dur["label"]]
        raw = cached_fetch(ticker, warm, None, None)
        cutoff = raw.index[-1] - pd.Timedelta(days=days) if days else raw.index[0]
    ind = su.add_indicators(raw)
    ind = ind[ind.index >= cutoff]
    if len(ind) < 5:
        raise ValueError("Not enough data in this duration. Choose a longer duration.")
    return ind


def duration_picker(key, default="1 Year"):
    options = list(DURATIONS) + ["Custom range"]
    label = st.selectbox("Duration", options, index=options.index(default), key=key + "_dur")
    if label == "Custom range":
        c1, c2 = st.columns(2)
        start = c1.date_input("From", dt.date.today() - dt.timedelta(days=365), key=key + "_from")
        end = c2.date_input("To", dt.date.today(), key=key + "_to")
        return {"label": label, "start": start, "end": end}
    return {"label": label, "start": None, "end": None}


def asset_picker(key):
    """Category + search box + result list + online search. Returns a Yahoo symbol."""
    c1, c2 = st.columns([1, 2])
    cat = c1.selectbox("Category", md.ALL_CATEGORIES, key=key + "_cat")
    query = c2.text_input("Search by name or symbol", key=key + "_q",
                          placeholder="e.g. Reliance, bitcoin, gold, nifty, AAPL")
    options = {f"{n}  ({s})": s for n, s, _ in md.search_catalog(query, cat)}
    ticker = "RELIANCE.NS"
    if options:
        label = st.selectbox(f"Results ({len(options)})", list(options), key=key + "_sel")
        ticker = options[label]
    else:
        st.info("Nothing in the built-in list matched. Use the online search below.")

    with st.expander("Can't find it? Search any symbol on Yahoo Finance"):
        q2 = st.text_input("Company name or symbol (this overrides the list above)", key=key + "_q2",
                           placeholder="e.g. Zomato, TATAMOTORS.NS, ^NSEI, ETH-USD")
        if q2.strip():
            found = cached_search(q2.strip())
            if found:
                pick = st.selectbox("Online results", [f for f, _ in found], key=key + "_online")
                ticker = dict(found)[pick]
            else:
                ticker = q2.strip().upper()
                st.caption(f"Online search returned nothing; trying '{ticker}' directly.")
    return ticker


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.session_state.setdefault("page", "Home")
st.sidebar.markdown("## 📈 Trend Predictor")
st.sidebar.caption("Learn, analyse and understand markets")
st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.divider()
st.sidebar.caption("Educational project. Not financial advice. Markets carry risk.")


# ============================================================================
# PAGE: HOME
# ============================================================================
def page_home():
    st.markdown("""
    <div class="hero">
      <span class="pill">Learn</span><span class="pill">Analyse</span><span class="pill">Predict</span><span class="pill">Stay Safe</span>
      <h1>Intelligent Stock Market Trend Predictor</h1>
      <p>An all-in-one place for beginners and traders: live prices for stocks, indices, crypto, commodities and forex,
      technical charts, an ML trend predictor, a full trading course and a safety center.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Market snapshot")
    rows = snapshot()
    if rows:
        for i in range(0, len(rows), 4):
            cols = st.columns(4)
            for col, (name, last, chg) in zip(cols, rows[i:i + 4]):
                col.metric(name, f"{last:,.2f}", f"{chg:+.2f}%")
        st.caption("Source: Yahoo Finance. Prices can be delayed by about 15 minutes. Last change = vs previous close.")
    else:
        st.warning("Could not load market data right now. Check your internet connection.")

    st.subheader("Where do you want to start?")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("### I am a complete beginner")
            st.write("Follow the 11-module course, take the quiz and learn how to open an account safely.")
            st.button("Open the course", key="h1", on_click=go_to, args=("Learn Trading",))
    with c2:
        with st.container(border=True):
            st.markdown("### I want to check a market")
            st.write("Search any stock, index, crypto, gold or currency. See prices, news and technical charts.")
            st.button("Explore markets", key="h2", on_click=go_to, args=("Market Explorer",))
    with c3:
        with st.container(border=True):
            st.markdown("### I want to avoid losing money")
            st.write("Golden rules, scam checklist, position-size and SIP calculators.")
            st.button("Open Safety Center", key="h3", on_click=go_to, args=("Safety Center",))

    st.subheader("What you can do here")
    f1, f2 = st.columns(2)
    f1.markdown(
        "- **Market Explorer:** search any asset, compare, read news\n"
        "- **Technical Analysis:** candlesticks, MA, RSI, MACD, Bollinger, volume, drawdown\n"
        "- **Predict Trend:** next-day UP/DOWN estimate with honest accuracy checks")
    f2.markdown(
        "- **Learn Trading:** course, quiz, glossary, free resources\n"
        "- **Broker & App Guide:** Groww, Angel One, Zerodha, Upstox and how to choose\n"
        "- **Safety Center:** rules, scam signs, calculators")
    st.warning(lc.DISCLAIMER)


# ============================================================================
# PAGE: MARKET EXPLORER
# ============================================================================
def page_market():
    st.title("Market Explorer")
    st.caption("Search stocks, indices, crypto, commodities, forex and ETFs.")
    ticker = asset_picker("mx")
    dur = duration_picker("mx")
    try:
        df = get_prices(ticker, dur)
    except Exception as e:
        st.error(f"Could not load data for {ticker}: {e}")
        return

    cs = md.currency_symbol(ticker)
    stats = md.risk_metrics(df)
    st.subheader(f"{md.name_of(ticker)}  ({ticker})")
    m = st.columns(5)
    m[0].metric("Last price", money(stats["last"], cs), f"{stats['change_pct']:+.2f}% (1 day)")
    m[1].metric(f"Return ({dur['label']})", f"{stats['period_return']:+.1f}%")
    m[2].metric("52-week high", money(stats["high_52w"], cs))
    m[3].metric("52-week low", money(stats["low_52w"], cs))
    m[4].metric("Risk level", stats["risk_level"], f"volatility {stats['volatility']:.0f}%/yr", delta_color="off")

    t_price, t_about, t_news, t_compare, t_data = st.tabs(["Price & Stats", "About", "News", "Compare", "Data"])

    with t_price:
        st.plotly_chart(viz.price_line_chart(df, f"{ticker} - Closing Price"))
        if df["Volume"].sum() > 0:
            st.plotly_chart(viz.volume_chart(df, ticker))
        st.plotly_chart(viz.drawdown_chart(df, ticker))
        st.info(f"In this period the biggest fall from a peak was **{stats['max_drawdown']:.1f}%**. "
                "Ask yourself honestly: could I stay calm if my money fell by this much?")

    with t_about:
        info = cached_info(ticker)
        if not info:
            st.info("Company details are not available for this symbol (common for indices, forex and some funds).")
        else:
            st.markdown(f"### {info.get('longName', md.name_of(ticker))}")
            a, b, c = st.columns(3)
            a.metric("Sector", info.get("sector", "-"))
            b.metric("Industry", info.get("industry", "-"))
            c.metric("Market cap", md.human_number(info.get("marketCap")))
            a, b, c = st.columns(3)
            a.metric("P/E ratio", f"{info['trailingPE']:.1f}" if "trailingPE" in info else "-")
            b.metric("P/B ratio", f"{info['priceToBook']:.1f}" if "priceToBook" in info else "-")
            c.metric("Exchange", info.get("exchange", "-"))
            if "longBusinessSummary" in info:
                st.write(info["longBusinessSummary"])
            st.caption("Learn what P/E, P/B and market cap mean in Learn Trading > Glossary.")

    with t_news:
        news = cached_news(ticker)
        if not news:
            st.info("No recent headlines found for this symbol.")
        for n in news:
            st.markdown(f"**[{n['title']}]({n['link']})**  \n<span class='small'>{n['source']} {n['time']}</span>",
                        unsafe_allow_html=True)
        st.caption("News is shown for information only. Headlines can be misleading; never trade on one headline.")

    with t_compare:
        names = {n: s for items in md.CATALOG.values() for n, s in items.items()}
        default = [md.name_of(ticker)] if md.name_of(ticker) in names else []
        default += [d for d in ["NIFTY 50"] if d not in default]
        picks = st.multiselect("Choose up to 5 assets to compare", list(names), default=default[:2], max_selections=5)
        if picks:
            series = {}
            for p in picks:
                try:
                    series[p] = get_prices(names[p], dur)["Close"]
                except Exception:
                    st.caption(f"Skipped {p} (no data).")
            if series:
                st.plotly_chart(viz.compare_chart(series))
                st.caption("Every line starts at 100, so you can compare growth fairly across different price levels.")

    with t_data:
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].round(2).sort_index(ascending=False))
        st.download_button("Download CSV", df[["Open", "High", "Low", "Close", "Volume"]].to_csv().encode("utf-8"),
                           file_name=f"{su.safe_name(ticker)}_prices.csv", mime="text/csv")
    footer()


# ============================================================================
# PAGE: TECHNICAL ANALYSIS
# ============================================================================
def page_technical():
    st.title("Technical Analysis")
    st.caption("Charts and indicators for any asset. Indicators describe the past; they do not guarantee the future.")
    ticker = asset_picker("ta")
    dur = duration_picker("ta", "6 Months")
    o1, o2, o3 = st.columns(3)
    as_line = o1.radio("Chart type", ["Candlestick", "Line"], horizontal=True) == "Line"
    show_ma = o2.checkbox("Moving averages (MA20, MA50)", value=True)
    show_bb = o3.checkbox("Bollinger Bands", value=True)
    try:
        df = get_prices(ticker, dur)
    except Exception as e:
        st.error(f"Could not load data for {ticker}: {e}")
        return

    st.plotly_chart(viz.candlestick_chart(df, f"{md.name_of(ticker)} ({ticker})", show_ma, show_bb, as_line))
    if df["Volume"].sum() > 0:
        st.plotly_chart(viz.volume_chart(df, ticker))
    st.plotly_chart(viz.rsi_chart(df, ticker))
    st.plotly_chart(viz.macd_chart(df, ticker))
    st.plotly_chart(viz.drawdown_chart(df, ticker))

    st.subheader("Signal summary (educational)")
    try:
        rows, overall = md.signal_summary(df)
        icon = {1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"}
        st.dataframe(pd.DataFrame([(r[0], r[1], icon[r[2]]) for r in rows],
                                  columns=["Indicator", "What it shows", "Lean"]))
        st.markdown(f"**Overall:** {overall}")
    except Exception:
        st.info("Not enough data to compute the signal summary. Choose a longer duration.")
    stats = md.risk_metrics(df)
    st.markdown(f"**Risk check:** {stats['risk_level']} (volatility about {stats['volatility']:.0f}% per year, "
                f"worst fall in this period {stats['max_drawdown']:.1f}%).")
    st.warning("**Use with caution:** this summary only reads past prices. It is not a buy or sell recommendation. "
               "Never invest because of one indicator, always set a stop-loss, risk only 1-2% of your capital per trade, "
               "and invest only money you can afford to lose.")
    with st.expander("How to read these charts"):
        st.markdown(
            "- **Candles:** green = closed higher than it opened, red = lower.\n"
            "- **MA20 / MA50:** price above both = uptrend, below both = downtrend.\n"
            "- **Bollinger Bands:** wide bands = high volatility.\n"
            "- **RSI:** above 70 overbought, below 30 oversold (not automatic sell/buy).\n"
            "- **MACD:** above signal line = positive momentum.\n"
            "- **Drawdown:** how far the price is below its previous peak.")
    footer()


# ============================================================================
# PAGE: PREDICT TREND
# ============================================================================
def page_predict():
    st.title("Predict Trend")
    st.warning("**Please read:** this estimates the chance that the **next trading day** closes higher than today, using only "
               "past prices. Short-term direction is very hard to predict (about 50-58% is a normal honest result). "
               "It cannot see news, results or global events. Do not trade only because of this number.")
    ticker = asset_picker("pr")
    name = md.name_of(ticker)

    c1, c2, c3 = st.columns([1, 1, 1])
    hist = c1.selectbox("Training history", ["2y", "5y", "10y", "max"], index=1,
                        help="More history gives the models more examples (usually better). 'max' uses all available data.")
    tf_ok = importlib.util.find_spec("tensorflow") is not None
    use_lstm = c2.checkbox("Include LSTM (slower)", value=False, disabled=not tf_ok,
                           help="Deep-learning model. Needs TensorFlow.")
    if not tf_ok:
        c2.caption("TensorFlow not installed: LSTM skipped.")
    c3.write("")
    if c3.button(f"Train / retrain models for {ticker}"):
        with st.spinner("Training models on past data... this can take a minute"):
            try:
                su.train_and_save(ticker, hist, use_lstm=use_lstm and tf_ok)
                st.session_state.pop("pred", None)
                st.success("Training complete.")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if not su.models_exist(ticker):
        st.info("No trained model for this asset yet. Click **Train / retrain models** first.")
        footer()
        return

    res = su.load_results(ticker)
    st.caption(f"Models trained on {res['n_train'] + res['n_test']} trading days "
               f"({res['start_date']} to {res['end_date']}), history setting: {res['period']}.")

    if st.button("Predict next trading day"):
        with st.spinner("Loading latest data and models..."):
            try:
                st.session_state["pred"] = (ticker, su.predict_tomorrow(ticker))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    saved = st.session_state.get("pred")
    if saved and saved[0] == ticker:
        out = saved[1]
        trend, conf = su.to_trend(out["ens_prob_up"])
        msg = f"{name} - Tomorrow: {trend} Trend (Confidence {conf}%)"
        if trend == "UP":
            st.success(f"## {msg}")
        else:
            st.error(f"## {msg}")
        st.caption(f"Meaning of confidence: {su.confidence_label(conf)}. Based on data up to {out['last_date']} "
                   f"(last close {money(out['last_close'], md.currency_symbol(ticker))}). "
                   "Headline = average of all models (ensemble).")

        cols = st.columns(3)
        for col, (label, key) in zip(cols, [("RandomForest", "rf_prob_up"), ("GradientBoosting", "gb_prob_up"),
                                            ("LSTM", "lstm_prob_up")]):
            p = out[key]
            if p is None:
                col.metric(label, "Not used")
            else:
                t, c = su.to_trend(p)
                col.metric(label, f"{t} ({c}%)", f"P(UP) = {p:.2f}", delta_color="off")

        ens = res["models"]["ensemble"]["metrics"]["Accuracy"]
        base = res["baseline_accuracy"]
        st.markdown(f"**Reality check:** on past data the model had never seen, the ensemble was right **{ens * 100:.1f}%** "
                    f"of the time. A coin flip gives 50%, and always guessing the more common direction gives {base * 100:.1f}%.")
        if ens <= base:
            st.error("On the test period this model did **not** beat the simple baseline. Treat today's prediction as unreliable.")
        st.subheader("Why? The indicators the RandomForest relied on")
        st.plotly_chart(viz.feature_importance_chart(res["importance"]))
        st.warning("**Use with caution and your own judgement.** Not financial advice. Risk only money you can afford to lose, "
                   "use a stop-loss, and never put all your money in one asset.")
    footer()


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
def page_performance():
    st.title("Model Performance")
    tickers = su.trained_tickers()
    if not tickers:
        st.info("No models trained yet. Go to **Predict Trend** and train a model for any asset.")
        return
    ticker = st.selectbox("Trained asset", tickers, format_func=lambda s: f"{md.name_of(s)} ({s})")
    res = su.load_results(ticker)
    models = {MODEL_NAMES[k]: v for k, v in res["models"].items()}

    st.write(f"Training days: **{res['n_train']}** | Test days (most recent 20%): **{res['n_test']}** | "
             f"Period: {res['start_date']} to {res['end_date']}")
    table = pd.DataFrame({n: m["metrics"] for n, m in models.items()}).T
    st.dataframe(table)

    best = max(models.items(), key=lambda kv: kv[1]["metrics"]["Accuracy"])
    a, b, c = st.columns(3)
    a.metric("Best accuracy", f"{best[1]['metrics']['Accuracy'] * 100:.1f}%", best[0], delta_color="off")
    b.metric("Coin flip", "50.0%")
    c.metric("'Always guess common side'", f"{res['baseline_accuracy'] * 100:.1f}%")
    if best[1]["metrics"]["Accuracy"] > 0.55:
        st.success("The best model is above the 55% objective on the test period.")
    else:
        st.warning("The best model is at or below 55% on the test period. This is normal for next-day direction; "
                   "report it honestly.")

    st.plotly_chart(viz.model_comparison_bar({n: m["metrics"] for n, m in models.items()}))
    st.subheader("Walk-forward validation (stability check)")
    wf = res["walk_forward"]
    st.plotly_chart(viz.walk_forward_chart(wf))
    st.caption("Average: " + " | ".join(f"{k} {np.mean(v) * 100:.1f}%" for k, v in wf.items()) +
               ". If the lines jump around 50%, the model has no stable edge.")
    st.subheader("Confusion matrices")
    cols = st.columns(len(models))
    for col, (n, m) in zip(cols, models.items()):
        with col:
            st.pyplot(viz.confusion_matrix_figure(m["cm"], n))
    st.plotly_chart(viz.actual_vs_predicted_chart(res["dates"], res["y_test"],
                                                  {n: m["pred"] for n, m in models.items()}))
    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Accuracy:** share of days predicted correctly.\n"
            "- **Precision:** of days predicted UP, how many were really UP.\n"
            "- **Recall:** of days that were really UP, how many were caught.\n"
            "- **F1:** balance of precision and recall.\n"
            "- **Confusion matrix:** shows how the model is wrong (e.g. always saying UP).\n"
            "- **Walk-forward:** trains on the past and tests on the next block several times; more reliable than one split.\n"
            "- The test set is small, so numbers change from run to run.")
    footer()


# ============================================================================
# PAGE: LEARN TRADING
# ============================================================================
def page_learn():
    st.title("Learn Trading from Scratch")
    st.caption("A free step-by-step manual: money basics, markets, charts, risk and psychology.")
    t_course, t_quiz, t_gloss, t_res = st.tabs(["Course", "Quiz", "Glossary", "Free Resources"])

    with t_course:
        titles = [m["title"] for m in lc.COURSE]
        pick = st.selectbox("Choose a module (go in order if you are new)", titles)
        mod = lc.COURSE[titles.index(pick)]
        st.markdown(f"<span class='pill'>{mod['level']}</span>", unsafe_allow_html=True)
        for i, (head, body) in enumerate(mod["lessons"]):
            with st.expander(f"{i + 1}. {head}", expanded=(i == 0)):
                st.markdown(body)
        st.success("**Key takeaways:** " + "  |  ".join(mod["takeaways"]))
        st.info("Next: try the idea on a real chart in **Technical Analysis**, then test yourself in the **Quiz** tab.")

    with t_quiz:
        st.write(f"{len(lc.QUIZ)} questions. Submit to see your score and explanations.")
        with st.form("quiz_form"):
            answers = []
            for i, (q, opts, _, _) in enumerate(lc.QUIZ):
                answers.append(st.radio(f"{i + 1}. {q}", opts, index=None, key=f"quiz_{i}"))
            submitted = st.form_submit_button("Submit quiz")
        if submitted:
            score = sum(1 for a, (_, opts, ci, _) in zip(answers, lc.QUIZ) if a == opts[ci])
            st.subheader(f"Score: {score} / {len(lc.QUIZ)}")
            for i, (a, (q, opts, ci, why)) in enumerate(zip(answers, lc.QUIZ)):
                ok = a == opts[ci]
                st.markdown(f"{'✅' if ok else '❌'} **{i + 1}. {q}**  \nCorrect: {opts[ci]}. {why}")

    with t_gloss:
        q = st.text_input("Search a term", placeholder="e.g. SIP, stop-loss, RSI")
        items = {k: v for k, v in lc.GLOSSARY.items() if q.lower() in k.lower() or q.lower() in v.lower()}
        if not items:
            st.info("No matching term.")
        for term in sorted(items):
            st.markdown(f"**{term}**: {items[term]}")

    with t_res:
        st.write("Trusted places to keep learning (verify details on the official sites):")
        for n, url, desc in lc.RESOURCES:
            st.markdown(f"- **{n}** (`{url}`): {desc}")
    footer()


# ============================================================================
# PAGE: BROKER & APP GUIDE
# ============================================================================
def page_brokers():
    st.title("Broker & App Guide")
    st.info("Neutral information only: no ranking, no referral links, no endorsement. Fees, features and offers change often, "
            "so always verify on the broker's official website.")
    t_b, t_open, t_check, t_safe = st.tabs(["Popular Apps", "Open an Account", "How to Choose", "Using Apps Safely"])
    with t_b:
        for name, kind, desc, good in lc.BROKERS:
            with st.expander(f"{name}  -  {kind}"):
                st.write(desc)
                st.caption(good)
        st.caption("Examples only; there are many other SEBI-registered brokers.")
    with t_open:
        for i, (h, b) in enumerate(lc.ACCOUNT_STEPS, start=1):
            st.markdown(f"**Step {i}: {h}**  \n{b}")
    with t_check:
        for item in lc.BROKER_CHECKLIST:
            st.markdown(f"- {item}")
    with t_safe:
        st.markdown(
            "- Install apps only from the links on the broker's official website.\n"
            "- Turn on 2-factor authentication; never share OTP, PIN or password.\n"
            "- Do not log in on public Wi-Fi or shared computers.\n"
            "- Start with a small amount and delivery orders.\n"
            "- Check every order screen (buy/sell, quantity, price, order type) before confirming.\n"
            "- Keep your phone number and email updated with the broker.\n"
            "- Do not enable F&O or commodity segments until you have learned and practised.")
    footer()


# ============================================================================
# PAGE: SAFETY CENTER
# ============================================================================
def page_safety():
    st.title("Safety Center")
    st.caption("How not to lose money: rules, scam signs and calculators.")
    t_rules, t_scam, t_calc, t_loss = st.tabs(["Golden Rules", "Scam Detector", "Calculators", "Loss Recovery"])

    with t_rules:
        for i, (rule, why) in enumerate(lc.GOLDEN_RULES, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {rule}**")
                st.caption(why)
        st.warning("No rule removes risk completely. These habits only reduce the chance of serious damage.")

    with t_scam:
        for sign, why in lc.SCAM_SIGNS:
            with st.expander(sign):
                st.write(why)
        st.error(lc.SCAM_ACTION)

    with t_calc:
        which = st.radio("Calculator", ["Position size", "Risk-reward", "SIP", "Lump-sum growth"], horizontal=True)
        if which == "Position size":
            a, b = st.columns(2)
            capital = a.number_input("Total trading capital", min_value=0.0, value=100000.0, step=1000.0)
            risk_pct = b.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            a, b = st.columns(2)
            entry = a.number_input("Entry price", min_value=0.01, value=500.0)
            stop = b.number_input("Stop-loss price", min_value=0.01, value=480.0)
            if stop >= entry:
                st.error("For a buy trade the stop-loss must be below the entry price.")
            else:
                risk_amt = capital * risk_pct / 100
                qty = math.floor(risk_amt / (entry - stop))
                st.metric("Maximum loss you accept", f"{risk_amt:,.2f}")
                st.metric("Quantity to buy", f"{qty}")
                st.metric("Total position value", f"{qty * entry:,.2f}")
                if qty * entry > capital:
                    st.warning("This position is bigger than your capital, so it would need borrowed money. "
                               "Use a smaller quantity or a tighter plan.")
        elif which == "Risk-reward":
            a, b, c = st.columns(3)
            entry = a.number_input("Entry", min_value=0.01, value=500.0)
            stop = b.number_input("Stop-loss", min_value=0.01, value=480.0)
            target = c.number_input("Target", min_value=0.01, value=550.0)
            if stop >= entry or target <= entry:
                st.error("For a buy trade: stop-loss < entry < target.")
            else:
                rr = (target - entry) / (entry - stop)
                st.metric("Risk : Reward", f"1 : {rr:.2f}")
                st.metric("Win rate needed to break even", f"{100 / (1 + rr):.0f}%")
                st.caption("Many traders look for at least 1 : 2, but your win rate also matters, and costs reduce profits.")
        elif which == "SIP":
            a, b, c = st.columns(3)
            monthly = a.number_input("Monthly investment", min_value=0.0, value=5000.0, step=500.0)
            rate = b.number_input("Expected yearly return (%)", min_value=0.0, max_value=30.0, value=12.0)
            years = c.number_input("Years", min_value=1, max_value=50, value=10)
            r, n = rate / 100 / 12, int(years) * 12
            fv = monthly * n if r == 0 else monthly * (((1 + r) ** n - 1) / r) * (1 + r)
            st.metric("You invest", f"{monthly * n:,.0f}")
            st.metric("Estimated value", f"{fv:,.0f}")
            st.caption("Illustration only. Real returns are not fixed and can be negative in some years.")
        else:
            a, b, c = st.columns(3)
            amount = a.number_input("Amount invested", min_value=0.0, value=100000.0, step=1000.0)
            rate = b.number_input("Expected yearly return (%)", min_value=0.0, max_value=30.0, value=10.0, key="lr")
            years = c.number_input("Years", min_value=1, max_value=50, value=10, key="ly")
            st.metric("Estimated value", f"{amount * (1 + rate / 100) ** int(years):,.0f}")
            st.caption("Illustration only. Not a promise.")

    with t_loss:
        st.write("Losses are harder to recover than they look:")
        losses = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        st.dataframe(pd.DataFrame({"If you lose (%)": losses,
                                   "You need to gain (%)": [round(x / (100 - x) * 100, 1) for x in losses]}))
        st.info("This is why protecting your capital with stop-loss and small position sizes matters more than chasing big profits.")
    footer()


# ============================================================================
# PAGE: TEAM DETAILS
# ============================================================================
def page_team():
    st.title("Team Details")
    with st.container(border=True):
        st.markdown("### Bhanu Kiran Bonam")
        st.markdown("**Role:** Project Developer (individual project)")
        st.markdown("**College:** DNR College of Engineering and Technology, Bhimavaram")
        st.markdown("**Course:** B.Tech Computer Science and Engineering, 2022-2026")
        st.markdown("**Work done:** data collection, feature engineering, model building, web application, "
                    "learning content and documentation")
    st.subheader("About the project")
    st.write("Intelligent Stock Market Trend Predictor is an educational final-year project. It predicts the next-day "
             "trend (UP / DOWN) using RandomForest, Gradient Boosting and LSTM models, and combines it with market "
             "data, technical charts, a trading course and a safety center for beginners.")
    st.markdown("**Tech stack:** Python, Streamlit, yfinance, pandas, numpy, ta, scikit-learn, TensorFlow/Keras (optional), "
                "Plotly, Matplotlib")
    st.warning(lc.DISCLAIMER)


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
{"Home": page_home, "Market Explorer": page_market, "Technical Analysis": page_technical,
 "Predict Trend": page_predict, "Model Performance": page_performance, "Learn Trading": page_learn,
 "Broker & App Guide": page_brokers, "Safety Center": page_safety, "Team Details": page_team,
 }[st.session_state["page"]]()
