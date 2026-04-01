import requests
import json

# Check market feed
try:
    response = requests.get('http://127.0.0.1:8000/api/market-feed')
    data = response.json()
    print(f'Market feed rows: {len(data.get("rows", []))}')
    if data.get('rows'):
        print('Sample market data:')
        for i, row in enumerate(data['rows'][:3]):
            print(f'  {i+1}. {row.get("symbol", "N/A")}: {row.get("ltp", "N/A")} ({row.get("change_pct", "N/A")}%)')
except Exception as e:
    print(f'Error getting market feed: {e}')

# Check signals
try:
    response = requests.get('http://127.0.0.1:8000/api/signals')
    data = response.json()
    print(f'\nSignals count: {len(data.get("signals", []))}')
    if data.get('signals'):
        print('Recent signals:')
        for i, signal in enumerate(data['signals'][:3]):
            print(f'  {i+1}. {signal.get("symbol", "N/A")} {signal.get("final_action", "N/A")} @ {signal.get("entry_price", "N/A")}')
except Exception as e:
    print(f'Error getting signals: {e}')
