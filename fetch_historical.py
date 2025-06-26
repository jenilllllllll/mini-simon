import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import credentials as cd

# === Setup ===
client_id = cd.client_id
access_token = open('access.txt', 'r').read()
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

# === Config ===
nifty_50_stocks = [
   "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ",
   "NSE:KOTAKBANK-EQ", "NSE:LT-EQ", "NSE:ITC-EQ", "NSE:SBIN-EQ", "NSE:HINDUNILVR-EQ",
   "NSE:AXISBANK-EQ", "NSE:BAJFINANCE-EQ", "NSE:ASIANPAINT-EQ", "NSE:MARUTI-EQ", "NSE:SUNPHARMA-EQ",
   "NSE:TITAN-EQ", "NSE:WIPRO-EQ", "NSE:ULTRACEMCO-EQ", "NSE:NESTLEIND-EQ", "NSE:POWERGRID-EQ",
   "NSE:BAJAJFINSV-EQ", "NSE:TECHM-EQ", "NSE:NTPC-EQ", "NSE:GRASIM-EQ", "NSE:JSWSTEEL-EQ",
   "NSE:HCLTECH-EQ", "NSE:TATAMOTORS-EQ", "NSE:DRREDDY-EQ", "NSE:CIPLA-EQ", "NSE:ONGC-EQ",
   "NSE:HDFCLIFE-EQ", "NSE:DIVISLAB-EQ", "NSE:HEROMOTOCO-EQ", "NSE:BRITANNIA-EQ", "NSE:BPCL-EQ",
   "NSE:COALINDIA-EQ", "NSE:ADANIENT-EQ", "NSE:ADANIPORTS-EQ", "NSE:INDUSINDBK-EQ", "NSE:BAJAJ-AUTO-EQ",
   "NSE:SHREECEM-EQ", "NSE:SBILIFE-EQ", "NSE:EICHERMOT-EQ", "NSE:TATACONSUM-EQ", "NSE:HINDALCO-EQ",
   "NSE:APOLLOHOSP-EQ", "NSE:ICICIPRULI-EQ", "NSE:TATASTEEL-EQ", "NSE:M&M-EQ", "NSE:BHARTIARTL-EQ"
]

timeframes = ["5", "15", "60", "240", "day"]
import os
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# === Fetch OHLCV ===
def fetch_ohlcv(symbol, timeframe):
    all_data = []
    today = datetime.utcnow()
    
    # Start from 10 years ago
    start_date = today - timedelta(days=365 * 10)
    max_days_per_call = 100
    delta = timedelta(days=max_days_per_call)

    while start_date < today:
        end_date = min(start_date + delta, today)
        payload = {
            "symbol": symbol,
            "resolution": timeframe,
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }

        try:
            response = fyers.history(payload)
            candles = response.get("candles", [])
            all_data.extend(candles)
        except Exception as e:
            print(f"❌ Error fetching {symbol} {timeframe} from {start_date.date()} to {end_date.date()}: {e}")
        
        start_date += delta + timedelta(days=1)  # move to next window

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s') + timedelta(hours=5, minutes=30)
    df.drop(columns='timestamp', inplace=True)

    if timeframe != "day":
        df = df[df['date'].dt.time.between(datetime.strptime("09:15", "%H:%M").time(),
                                           datetime.strptime("15:30", "%H:%M").time())]

    return df

# === Feature Engineering ===
def add_features(df):
    df['body'] = abs(df['close'] - df['open'])
    df['wick_top'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['wick_bottom'] = df[['close', 'open']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 1e-6)
    df['is_bullish'] = df['close'] > df['open']

    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['vol_ma20']

    df['swing_high'] = (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
    df['swing_low'] = (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])

    df['hour'] = df['date'].dt.hour
    df['session'] = np.select(
        [
            df['hour'] < 9,
            df['hour'].between(9, 11),
            df['hour'].between(11, 13),
            df['hour'] >= 13
        ],
        [
            'pre_market',
            'morning',
            'noon',
            'closing'
        ],
        default='unknown'
    )
    return df

# === Bias Function ===
def get_bias(higher_df, current_time):
    past = higher_df[higher_df['date'] < current_time].tail(3)
    if len(past) < 3:
        return 'neutral'
    highs = past['high'].values
    lows = past['low'].values
    if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
        return 'bullish'
    elif highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
        return 'bearish'
    return 'neutral'

# === Master Loop ===
for tf in timeframes:
    print(f"\n=== Timeframe: {tf} ===")
    for symbol in nifty_50_stocks:
        name = symbol.split(":")[1].replace("-EQ", "")
        df = fetch_ohlcv(symbol, tf)
        if df.empty:
            print(f"⚠️ No data for {name} at {tf} timeframe.")
            continue

        df = add_features(df)

        # Apply bias only if tf != day
        if tf != "day":
            higher_tf_map = {"5": "60", "15": "60", "60": "240", "240": "day"}
            higher_tf = higher_tf_map.get(tf)
            if higher_tf:
                higher_path = os.path.join(base_path, higher_tf, f"{name}.csv")
                if os.path.exists(higher_path):
                    higher_df = pd.read_csv(higher_path)
                    higher_df['date'] = pd.to_datetime(higher_df['date'])
                    df['date'] = pd.to_datetime(df['date'])
                    df['bias'] = df['date'].apply(lambda x: get_bias(higher_df, x))

        folder = os.path.join(base_path, tf)
        os.makedirs(folder, exist_ok=True)
        label = "daily" if tf == "day" else f"{tf}-min"
        filepath = os.path.join(folder, f"{name}_{label}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
