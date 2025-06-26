import os
import pandas as pd
from fyers_apiv3 import fyersModel
import datetime as dt
import time
import credentials as cd

# -----------------------------
# ⚙ Setup
# -----------------------------
client_id = cd.client_id
access_token = open('access.txt', 'r').read()
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

import os
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
timeframes = ['240', 'day']
nifty_50_stocks = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "LT", "ITC", "SBIN", "HINDUNILVR",
    "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "BAJAJFINSV", "TECHM", "NTPC", "GRASIM", "JSWSTEEL", "HCLTECH", "TATAMOTORS", "DRREDDY", "CIPLA",
    "ONGC", "HDFCLIFE", "DIVISLAB", "HEROMOTOCO", "BRITANNIA", "BPCL", "COALINDIA", "ADANIENT", "ADANIPORTS",
    "INDUSINDBK", "BAJAJ-AUTO", "SHREECEM", "SBILIFE", "EICHERMOT", "TATACONSUM", "HINDALCO", "APOLLOHOSP",
    "ICICIPRULI", "TATASTEEL", "M&M", "BHARTIARTL"
]

# -----------------------------
# 📥 Helper: Fetch day data
# -----------------------------
def fetch_day_data(symbol_code, filename):
    all_data = []
    count = 0
    to_date = dt.datetime.now()
    while count < 15:  # Cap: 15*700 = ~10.5 years max
        from_date = to_date - dt.timedelta(days=700)
        data = {
            "symbol": f"NSE:{symbol_code}-EQ",
            "resolution": "D",
            "date_format": "1",
            "range_from": from_date.strftime('%Y-%m-%d'),
            "range_to": to_date.strftime('%Y-%m-%d'),
            "cont_flag": "1"
        }
        try:
            response = fyers.history(data)
            candles = response.get("candles", [])
            if not candles:
                break
            all_data = candles + all_data
            to_date = from_date
            count += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error fetching {symbol_code} DAY: {e}")
            break

    if all_data:
        df = pd.DataFrame(all_data, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"], unit='s')
        df.to_csv(filename, index=False)
        print(f"✅ Fetched missing day data: {symbol_code}")
    else:
        print(f"⚠ No data found for {symbol_code} (DAY)")


# -----------------------------
# 🔄 Main Loop
# -----------------------------
for tf in timeframes:
    print(f"\n=== Checking timeframe: {tf} ===")
    tf_path = os.path.join(base_path, tf)
    os.makedirs(tf_path, exist_ok=True)

    for symbol in nifty_50_stocks:
        file_path = os.path.join(tf_path, f"{symbol}_{tf}-min.csv")

        # 🧾 If Day file is missing, re-fetch it
        if tf == 'day' and not os.path.exists(file_path):
            print(f"🟡 Missing: {file_path}")
            fetch_day_data(symbol, file_path)
            continue

        # 🧪 Skip if file missing (other TFs)
        if not os.path.exists(file_path):
            print(f"⚠ Missing file: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            if len(df) < 100:
                print(f"🔻 Skipping short data: {symbol} (Only {len(df)} rows in {tf})")
                continue

            # ✨ Example: Show first row
            print(f"✅ {symbol} ({tf}): {len(df)} rows loaded")

        except Exception as e:
            print(f"❌ Error reading {symbol} ({tf}): {e}")
