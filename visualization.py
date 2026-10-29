"""
visualization.py  --  Module 4 : Visualization (Plotly + Matplotlib)
=====================================================================
Every function takes data and returns a figure. The Streamlit app (app.py)
and the notebook both reuse these functions.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def candlestick_chart(df, ticker):
    """Candlestick chart + MA20, MA50 and Bollinger Bands.
    `df` must be the output of stock_utils.add_indicators()."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                             line=dict(width=1.5, color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                             line=dict(width=1.5, color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_high"], name="BB upper",
                             line=dict(width=1, dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB lower",
                             line=dict(width=1, dash="dot", color="gray")))
    fig.update_layout(title=f"{ticker} - Candlestick with MA20, MA50 and Bollinger Bands",
                      xaxis_rangeslider_visible=False, height=550,
                      yaxis_title="Price (INR)")
    return fig


def rsi_chart(df, ticker):
    """RSI line with the 70 (overbought) and 30 (oversold) levels."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f"{ticker} - RSI (14)", height=300,
                      yaxis=dict(range=[0, 100]))
    return fig


def macd_chart(df, ticker):
    """MACD line, signal line and histogram."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
    fig.update_layout(title=f"{ticker} - MACD", height=300)
    return fig


def actual_vs_predicted_chart(dates, y_true, predictions):
    """
    Step chart of actual trend vs the trend predicted by each model
    on the TEST period. predictions = {"RandomForest": [...], "LSTM": [...]}
    1 = UP, 0 = DOWN.
    """
    series = {"Actual": y_true}
    series.update(predictions)
    fig = make_subplots(rows=len(series), cols=1, shared_xaxes=True,
                        subplot_titles=list(series.keys()))
    for i, (name, values) in enumerate(series.items(), start=1):
        fig.add_trace(go.Scatter(x=dates, y=values, mode="lines",
                                 line_shape="hv", name=name), row=i, col=1)
        fig.update_yaxes(tickvals=[0, 1], ticktext=["DOWN", "UP"], row=i, col=1)
    fig.update_layout(title="Actual vs Predicted Trend (Test Period)",
                      height=250 * len(series), showlegend=False)
    return fig


def model_comparison_bar(metrics_by_model):
    """Grouped bar chart. metrics_by_model = {"RandomForest": {"Accuracy": ..}, "LSTM": {...}}"""
    fig = go.Figure()
    for model_name, metrics in metrics_by_model.items():
        fig.add_trace(go.Bar(name=model_name, x=list(metrics.keys()),
                             y=list(metrics.values())))
    fig.update_layout(barmode="group", title="Model Comparison",
                      yaxis=dict(range=[0, 1], title="Score"), height=400)
    return fig


def feature_importance_chart(importance):
    """Horizontal bar chart of RandomForest feature importance (explainability)."""
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                           orientation="h"))
    fig.update_layout(title="RandomForest - Which indicators mattered most?",
                      height=450, xaxis_title="Importance")
    return fig


def confusion_matrix_figure(cm, title):
    """Matplotlib confusion matrix. Rows = actual, columns = predicted."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["DOWN", "UP"])
    ax.set_yticklabels(["DOWN", "UP"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.tight_layout()
    return fig
