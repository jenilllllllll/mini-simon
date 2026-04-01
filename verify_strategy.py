
import json
import os
import sys

def verify_trade_log():
    log_file = r"c:\Drive data\Bed Room Trader\mini-simon\trade_log.json"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        trades = data.get("Trades", [])
        if not trades:
            print("No trades found in log yet.")
            return

        print(f"Found {len(trades)} trades.")
        for i, trade in enumerate(trades[-5:]): # Check last 5 trades
            strategy = trade.get("strategy", "MISSING")
            print(f"Trade {i+1}: Symbol={trade.get('symbol')}, Strategy={strategy}")
            
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    verify_trade_log()
