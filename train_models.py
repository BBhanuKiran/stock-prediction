"""
train_models.py - optional: pre-train models for a few assets from the command line
(The web app can also train any asset from the 'Predict Trend' page.)
Usage:  python train_models.py
Creates: models/<SYMBOL>_rf.pkl, _gb.pkl, _scaler.pkl, _lstm.h5 (if TensorFlow), _results.pkl
"""
import stock_utils as su

ASSETS = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "^NSEI"]   # add any Yahoo Finance symbols here

for ticker in ASSETS:
    print(f"\n=== Training for {ticker} ===")
    try:
        results = su.train_and_save(ticker, period="5y")
        for key, m in results["models"].items():
            print(f"{key:9s}", m["metrics"])
        print("Baseline (always guess commoner side):", results["baseline_accuracy"])
    except Exception as e:
        print(f"Failed for {ticker}: {e}")

print("\nDone. Now run:  streamlit run app.py")
