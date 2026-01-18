import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os

def generate_sample_data(strategy_name: str, n_symbols: int = 10, n_days: int = 5):
    """Generate sample strategy data for testing."""
    # Base symbols to use
    base_symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", 
                   "BHARTIARTL", "TATAMOTORS", "HINDUNILVR", "ITC", "SBIN"]
    symbols = [f"NSE:{s}-EQ" for s in base_symbols[:n_symbols]]
    
    # Generate timestamps (last n_days, 15-min intervals during market hours)
    start_date = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0) - timedelta(days=n_days)
    timestamps = []
    current = start_date
    while current <= datetime.now():
        if current.weekday() < 5:  # Weekdays only
            if time(9, 15) <= current.time() <= time(15, 30):  # Market hours
                timestamps.append(current)
        current += timedelta(minutes=15)
    
    # Create sample data
    data = []
    for symbol in symbols:
        for ts in timestamps:
            # Randomly decide if this is a signal (30% chance)
            if np.random.random() > 0.7:
                signal = np.random.choice(['BUY', 'SELL'], p=[0.6, 0.4])
                score = np.random.uniform(0.3, 1.0)  # Score between 0.3 and 1.0
                price = np.random.uniform(100, 5000)
                volume = int(np.random.uniform(1000, 1000000))
                
                data.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'price': round(price, 2),
                    'signal': signal,
                    'score': round(score, 2),
                    'volume': volume
                })
    
    # Create DataFrame and save to CSV
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values(['symbol', 'timestamp'])
        df.to_csv(f"{strategy_name}.csv", index=False)
        print(f"Generated {len(df)} records for {strategy_name}")
    else:
        print(f"No data generated for {strategy_name}")

if __name__ == "__main__":
    # Create test data for all strategies
    strategies = [
        'vol_spike',
        'body_imbalance',
        'order_block',
        'stock_burner'
    ]
    
    for strat in strategies:
        generate_sample_data(strat, n_symbols=5, n_days=2)
    
    print("\nTest data generation complete. You can now run strategy_aggregator.py")
