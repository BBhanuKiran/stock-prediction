"""
visualization.py  --  Module 4 : Visualization (Plotly + Matplotlib)
=====================================================================
Every function takes data and returns a figure. app.py and the notebook reuse them.
Colour theme: navy background, gold accents (premium look).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

GOLD, GREEN, RED, BLUE, GREY = "#C9A227", "#2ECC71", "#E74C3C", "#5DADE2", "#8A94A6"
TEXT = "#E8ECF4"


def _style(fig, height=None, title=None):
    """One common look for all Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=title)
    return fig


def price_line_chart(df, title="Price"):
    """Simple closing-price line with soft fill."""
    fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close",
                               line=dict(color=GOLD, width=2), fill="tozeroy",
                               fillcolor="rgba(201,162,39,0.08)"))
    fig.update_yaxes(rangemode="normal")
    return _style(fig, 420, title)


def candlestick_chart(df, ticker, show_ma=True, show_bb=True, as_line=False):
    """Candlestick (or line) with optional MA20, MA50 and Bollinger Bands.
    df must come from stock_utils.add_indicators()."""
    fig = go.Figure()
    if as_line:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=GOLD, width=2)))
    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                     close=df["Close"], name="Price",
                                     increasing_line_color=GREEN, decreasing_line_color=RED))
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(width=1.5, color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(width=1.5, color=BLUE)))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_high"], name="BB upper",
                                 line=dict(width=1, dash="dot", color=GREY)))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB lower",
                                 line=dict(width=1, dash="dot", color=GREY)))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, 540, f"{ticker} - Price Chart")


def volume_chart(df, ticker):
    """Volume bars coloured green (up day) or red (down day)."""
    colors = [GREEN if c >= o else RED for o, c in zip(df["Open"], df["Close"])]
    fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"))
    return _style(fig, 260, f"{ticker} - Volume")


def rsi_chart(df, ticker):
    """RSI with 70 (overbought) and 30 (oversold) lines."""
    fig = go.Figure(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)", line=dict(color=GOLD)))
    fig.add_hline(y=70, line_dash="dash", line_color=RED)
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN)
    fig.update_yaxes(range=[0, 100])
    return _style(fig, 280, f"{ticker} - RSI (14)")


def macd_chart(df, ticker):
    """MACD line, signal line and histogram."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram", marker_color=GREY))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color=BLUE)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color=GOLD)))
    return _style(fig, 280, f"{ticker} - MACD")


def drawdown_chart(df, ticker):
    """How far the price is below its previous peak (shows the pain of holding)."""
    dd = (df["Close"] / df["Close"].cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=df.index, y=dd, fill="tozeroy", name="Drawdown %",
                               line=dict(color=RED), fillcolor="rgba(231,76,60,0.25)"))
    fig.update_yaxes(ticksuffix="%")
    return _style(fig, 260, f"{ticker} - Drawdown (fall from previous peak)")


def compare_chart(series_dict):
    """Compare several assets: every line starts at 100 so growth is comparable.
    series_dict = {"Name": close_series}"""
    fig = go.Figure()
    for name, s in series_dict.items():
        s = s.dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=name, mode="lines"))
    fig.update_yaxes(title="Growth of 100")
    return _style(fig, 450, "Comparison (start = 100)")


def actual_vs_predicted_chart(dates, y_true, predictions):
    """Step chart: actual trend vs each model's predicted trend on the test period."""
    series = {"Actual": y_true}
    series.update(predictions)
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True, subplot_titles=list(series.keys()))
    for i, (name, values) in enumerate(series.items(), start=1):
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines", line_shape="hv", name=name,
                                 line=dict(color=GOLD if name == "Actual" else BLUE)), row=i, col=1)
        fig.update_yaxes(tickvals=[0, 1], ticktext=["DOWN", "UP"], row=i, col=1)
    fig.update_layout(showlegend=False)
    return _style(fig, 200 * len(series) + 80, "Actual vs Predicted Trend (Test Period)")


def model_comparison_bar(metrics_by_model):
    """Grouped bars. metrics_by_model = {"RandomForest": {"Accuracy": .., ...}, ...}"""
    fig = go.Figure()
    for name, metrics in metrics_by_model.items():
        fig.add_trace(go.Bar(name=name, x=list(metrics.keys()), y=list(metrics.values())))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], title="Score")
    return _style(fig, 400, "Model Comparison")


def walk_forward_chart(wf):
    """Accuracy of each walk-forward fold. wf = {"RandomForest": [..], ...}"""
    fig = go.Figure()
    for name, scores in wf.items():
        fig.add_trace(go.Scatter(x=[f"Fold {i + 1}" for i in range(len(scores))], y=scores,
                                 mode="lines+markers", name=name))
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY)
    fig.update_yaxes(range=[0.3, 0.8], title="Accuracy")
    return _style(fig, 340, "Walk-forward accuracy (dashed line = coin flip)")


def feature_importance_chart(importance):
    """Which indicators the RandomForest relied on most (explainability)."""
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=GOLD))
    return _style(fig, 480, "Which indicators mattered most?")


def confusion_matrix_figure(cm, title):
    """Matplotlib confusion matrix (rows = actual, columns = predicted)."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    fig.patch.set_alpha(0)
    ax.imshow(cm, cmap="YlOrBr")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["DOWN", "UP"], color=TEXT); ax.set_yticklabels(["DOWN", "UP"], color=TEXT)
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("Actual", color=TEXT)
    ax.set_title(title, color=TEXT)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.tight_layout()
    return fig
