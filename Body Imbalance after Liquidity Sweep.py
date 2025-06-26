import os
import pandas as pd

# === CONFIGURATION ===
input_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
output_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Body Imbalance after Liquidity Sweep")
timeframes = ['5', '15', '60', '240', 'day']

nifty_50_stocks = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "LT", "ITC", "SBIN", "HINDUNILVR",
    "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "BAJAJFINSV", "TECHM", "NTPC", "GRASIM", "JSWSTEEL", "HCLTECH", "TATAMOTORS", "DRREDDY", "CIPLA",
    "ONGC", "HDFCLIFE", "DIVISLAB", "HEROMOTOCO", "BRITANNIA", "BPCL", "COALINDIA", "ADANIENT", "ADANIPORTS",
    "INDUSINDBK", "BAJAJ-AUTO", "SHREECEM", "SBILIFE", "EICHERMOT", "TATACONSUM", "HINDALCO", "APOLLOHOSP",
    "ICICIPRULI", "TATASTEEL", "M&M", "BHARTIARTL"
]

# === STRATEGY FUNCTION ===
def apply_strategy(df):
    trades = []

    for i in range(1, len(df) - 1):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # Common Filters
        vol_confirm = next_row['volume_spike']
        is_valid_session = next_row['session'] in ['morning', 'afternoon']
        is_not_inside_bar = not ((next_row['high'] < curr['high']) and (next_row['low'] > curr['low']))

        # === LONG SETUP ===
        is_liquidity_sweep_down = curr['low'] < prev['low']
        is_bullish_imbalance = next_row['is_bullish'] and next_row['body_ratio'] > 0.5
        wick_rejection = next_row['wick_bottom'] > next_row['wick_top']

        if all([is_liquidity_sweep_down, is_bullish_imbalance, vol_confirm, is_valid_session, wick_rejection, is_not_inside_bar]):
            entry_price = next_row['close']
            stop_loss = next_row['low']
            take_profit = entry_price + 2 * (entry_price - stop_loss)
            risk = entry_price - stop_loss

            for j in range(i + 2, len(df)):
                future = df.iloc[j]
                if future['low'] <= stop_loss:
                    trades.append({
                        'entry_time': next_row['date'],
                        'exit_time': future['date'],
                        'direction': 'long',
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'result': 'loss',
                        'points': -risk
                    })
                    break
                elif future['high'] >= take_profit:
                    trades.append({
                        'entry_time': next_row['date'],
                        'exit_time': future['date'],
                        'direction': 'long',
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'result': 'win',
                        'points': 2 * risk
                    })
                    break

        # === SHORT SETUP ===
        is_liquidity_sweep_up = curr['high'] > prev['high']
        is_bearish_imbalance = not next_row['is_bullish'] and next_row['body_ratio'] > 0.5
        wick_rejection_short = next_row['wick_top'] > next_row['wick_bottom']

        if all([is_liquidity_sweep_up, is_bearish_imbalance, vol_confirm, is_valid_session, wick_rejection_short, is_not_inside_bar]):
            entry_price = next_row['close']
            stop_loss = next_row['high']
            take_profit = entry_price - 2 * (stop_loss - entry_price)
            risk = stop_loss - entry_price

            for j in range(i + 2, len(df)):
                future = df.iloc[j]
                if future['high'] >= stop_loss:
                    trades.append({
                        'entry_time': next_row['date'],
                        'exit_time': future['date'],
                        'direction': 'short',
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'result': 'loss',
                        'points': -risk
                    })
                    break
                elif future['low'] <= take_profit:
                    trades.append({
                        'entry_time': next_row['date'],
                        'exit_time': future['date'],
                        'direction': 'short',
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'result': 'win',
                        'points': 2 * risk
                    })
                    break

    return pd.DataFrame(trades)


# === MAIN BACKTEST LOOP ===
for tf in timeframes:
    for stock in nifty_50_stocks:
        try:
            input_file = os.path.join(input_base_path, tf, f"{stock}_{tf}-min.csv")
            if not os.path.exists(input_file):
                print(f"⛔ Missing: {input_file}")
                continue

            df = pd.read_csv(input_file, parse_dates=['date'])
            df.dropna(inplace=True)

            if len(df) < 100:
                print(f"⚠️ Skipping short data: {stock}-{tf}")
                continue

            trades_df = apply_strategy(df)

            if trades_df.empty:
                print(f"⚠️ No trades: {stock}-{tf}")
                continue

            # Summary calculation
            total_trades = len(trades_df)
            wins = trades_df[trades_df['result'] == 'win']
            losses = trades_df[trades_df['result'] == 'loss']
            win_rate = round(len(wins) / total_trades * 100, 2)
            avg_win = round(wins['points'].mean(), 2) if not wins.empty else 0
            avg_loss = round(losses['points'].mean(), 2) if not losses.empty else 0
            profit_factor = round(wins['points'].sum() / abs(losses['points'].sum()), 2) if not losses.empty else float('inf')

            # Output path
            output_dir = os.path.join(output_base_path, tf)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{stock}.xlsx")

            # Save Excel
            with pd.ExcelWriter(output_file) as writer:
                trades_df.to_excel(writer, sheet_name="Trades", index=False)
                pd.DataFrame([{
                    'Total Trades': total_trades,
                    'Wins': len(wins),
                    'Losses': len(losses),
                    'Win Rate (%)': win_rate,
                    'Avg Win': avg_win,
                    'Avg Loss': avg_loss,
                    'Profit Factor': profit_factor
                }]).to_excel(writer, sheet_name="Summary", index=False)

            print(f"✅ {stock}-{tf}: Trades={total_trades}, Win%={win_rate}, PF={profit_factor}")

        except Exception as e:
            print(f"❌ Error in {stock}-{tf}: {e}")
