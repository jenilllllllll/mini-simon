import os
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------------------

# 1. Module Name (Used for output file naming and folder structure)
MODULE_NAME = "LiquiditySweep"

# 2. Data & Output Paths
# Note: Please verify these paths are correct for your system.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# The data path has been updated to match the one specified in the prompt.
DATA_BASE_PATH = r"C:\Drive data\Bed Room Trader\mini-simon\Data"
OUTPUT_BASE_PATH = os.path.join(PROJECT_ROOT, "module", MODULE_NAME)

# 3. Timeframes to Backtest
# The script will map 'day' to the folder 'D' as requested.
TIMEFRAMES = ['240', 'day']

# 4. Stock Universe
NIFTY_50_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "LT", "ITC", "SBIN", "HINDUNILVR",
    "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "BAJAJFINSV", "TECHM", "NTPC", "GRASIM", "JSWSTEEL", "HCLTECH", "TATAMOTORS", "DRREDDY", "CIPLA",
    "ONGC", "HDFCLIFE", "DIVISLAB", "HEROMOTOCO", "BRITANNIA", "BPCL", "COALINDIA", "ADANIENT", "ADANIPORTS",
    "INDUSINDBK", "BAJAJ-AUTO", "SHREECEM", "SBILIFE", "EICHERMOT", "TATACONSUM", "HINDALCO", "APOLLOHOSP",
    "ICICIPRULI", "TATASTEEL", "M&M", "BHARTIARTL"
]

# -----------------------------------------------------------------------------
# --- STRATEGY LOGIC (PLACEHOLDER) ---
# -----------------------------------------------------------------------------

def run_strategy_backtest(df):
    """
    Executes the backtest for the given data.
    
    !!! IMPORTANT !!!
    This function contains a DUMMY placeholder strategy.
    You need to REPLACE this with your actual "Volume Spike + Liquidity Sweep" logic.
    
    The function must return a DataFrame representing the trade log.
    
    Args:
        df (pd.DataFrame): A DataFrame with OHLCV and pre-calculated indicators.
        
    Returns:
        pd.DataFrame: A trade log with columns:
                      ['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price']
    """
    # Placeholder Logic: Buy on a bullish candle if the previous was bearish. Exit after 5 bars.
    trades = []
    in_trade = False
    exit_bar_index = 0

    # Ensure date column is in datetime format
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    for i in range(1, len(df)):
        if in_trade and i >= exit_bar_index:
            # Exit trade
            trades[-1]['exit_time'] = df.at[i, 'date']
            trades[-1]['exit_price'] = df.at[i, 'open']
            in_trade = False

        if not in_trade:
            # Entry condition: bullish candle after a bearish one
            if df.at[i, 'is_bullish'] and not df.at[i-1, 'is_bullish']:
                trades.append({
                    'entry_time': df.at[i, 'date'],
                    'entry_price': df.at[i, 'close'],
                    'direction': 'Buy',
                    'exit_time': None, # To be filled upon exit
                    'exit_price': None # To be filled upon exit
                })
                in_trade = True
                exit_bar_index = i + 5 # Exit after 5 bars

    return pd.DataFrame(trades)

# -----------------------------------------------------------------------------
# --- PERFORMANCE CALCULATION & REPORTING ---
# -----------------------------------------------------------------------------

def calculate_performance_summary(trade_log_df):
    """Calculates performance metrics from a trade log."""
    if trade_log_df.empty:
        return pd.DataFrame(), pd.DataFrame([{
            'Total Trades': 0, 'Wins': 0, 'Losses': 0, 'Win Rate (%)': 0,
            'Avg Win': 0, 'Avg Loss': 0, 'Profit Factor': 0
        }])

    # Calculate points and result (Win/Loss)
    trade_log_df['points'] = np.where(
        trade_log_df['direction'] == 'Buy',
        trade_log_df['exit_price'] - trade_log_df['entry_price'],
        trade_log_df['entry_price'] - trade_log_df['exit_price']
    )
    trade_log_df['result'] = np.where(trade_log_df['points'] > 0, 'Win', 'Loss')

    # Calculate summary metrics
    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['result'] == 'Win']
    losses = trade_log_df[trade_log_df['result'] == 'Loss']
    
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = wins['points'].mean() if num_wins > 0 else 0
    avg_loss = losses['points'].mean() if num_losses > 0 else 0
    
    gross_profit = wins['points'].sum()
    gross_loss = abs(losses['points'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    summary_data = {
        'Total Trades': total_trades,
        'Wins': num_wins,
        'Losses': num_losses,
        'Win Rate (%)': f"{win_rate:.2f}",
        'Avg Win': f"{avg_win:.2f}",
        'Avg Loss': f"{avg_loss:.2f}",
        'Profit Factor': f"{profit_factor:.2f}"
    }
    
    return trade_log_df, pd.DataFrame([summary_data])

def save_results_to_excel(trade_log, summary, filepath):
    """Saves the trade log and performance summary to a two-sheet Excel file."""
    try:
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            trade_log.to_excel(writer, sheet_name='Trade Log', index=False)
            summary.to_excel(writer, sheet_name='Performance Summary', index=False)
            
            # Set the summary sheet to landscape
            workbook = writer.book
            summary_sheet = writer.sheets['Performance Summary']
            summary_sheet.set_landscape()
        print(f"✅ Successfully saved results to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving Excel file for {os.path.basename(filepath)}: {e}")
        print("   Please ensure you have 'xlsxwriter' installed: pip install xlsxwriter")

# -----------------------------------------------------------------------------
# --- MAIN EXECUTION LOOP ---
# -----------------------------------------------------------------------------

def main():
    """Main function to run the backtesting process."""
    for tf in TIMEFRAMES:
        # Decouple data source folder from output folder to handle mismatch
        data_tf_folder = tf  # Data source uses 'day'
        output_tf_folder = 'D' if tf == 'day' else tf # Output uses 'D' as per spec

        data_tf_path = os.path.join(DATA_BASE_PATH, data_tf_folder)
        output_tf_path = os.path.join(OUTPUT_BASE_PATH, output_tf_folder)
        os.makedirs(output_tf_path, exist_ok=True)
        
        print(f"\n{'='*20} Processing Timeframe: {tf.upper()} {'='*20}")

        for stock in NIFTY_50_STOCKS:
            # Use the correct file naming convention from your data source
            label = "daily" if tf == "day" else f"{tf}-min"
            data_file_path = os.path.join(data_tf_path, f"{stock}_{label}.csv")

            if not os.path.exists(data_file_path):
                # Add a fallback to check for the alternate naming convention (e.g., RELIANCE.csv)
                data_file_path_alt = os.path.join(data_tf_path, f"{stock}.csv")
                if not os.path.exists(data_file_path_alt):
                    print(f"- Skipping {stock}: Data file not found at {data_file_path} or {data_file_path_alt}")
                    continue
                else:
                    data_file_path = data_file_path_alt # Use the alternate path if found

            try:
                # 1. Load Data
                df = pd.read_csv(data_file_path)
                if len(df) < 100:
                    print(f"- Skipping {stock}: Insufficient data ({len(df)} rows)")
                    continue

                # 2. Run Backtest
                trade_log_df = run_strategy_backtest(df)

                # 3. Calculate Performance
                final_trade_log, summary_df = calculate_performance_summary(trade_log_df)

                # 4. Print Summary to Terminal
                print(f"\n--- Results for {stock} [{tf.upper()}] ---")
                print(summary_df.to_string(index=False))
                print("-" * 40)

                # 5. Save Results to Excel
                output_filename = f"{stock}-{output_tf_folder}-{MODULE_NAME}.xlsx"
                output_filepath = os.path.join(output_tf_path, output_filename)
                save_results_to_excel(final_trade_log, summary_df, output_filepath)

            except Exception as e:
                print(f"❌ An unexpected error occurred for {stock} [{tf.upper()}]: {e}")

if __name__ == "__main__":
    main()
