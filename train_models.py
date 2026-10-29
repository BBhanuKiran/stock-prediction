"""
train_models.py - trains and saves models for all stocks (run once before the app)
Usage:  python train_models.py
Creates: models/<TICKER>_rf.pkl, _scaler.pkl, _lstm.h5, _results.pkl
"""
import stock_utils as su

for ticker in su.STOCKS:
    print(f"\n=== Training for {ticker} ===")
    try:
        results = su.train_and_save(ticker, period="2y")
        print("RandomForest:", results["rf"]["metrics"])
        if results["lstm"]:
            print("LSTM        :", results["lstm"]["metrics"])
    except Exception as e:
        print(f"Failed for {ticker}: {e}")

print("\nDone. Now run:  streamlit run app.py")
