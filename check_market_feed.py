import requests

# Check what the dashboard is actually returning
try:
    response = requests.get('http://127.0.0.1:8000/api/market-feed')
    data = response.json()
    rows = data.get('rows', [])
    print(f'API Response: {len(rows)} rows')
    
    # Count by symbol type
    equity_symbols = []
    commodity_symbols = []
    other_symbols = []
    
    for row in rows:
        symbol = row.get('symbol', '')
        if symbol.startswith('MCX:'):
            commodity_symbols.append(symbol)
        elif ':' in symbol:
            equity_symbols.append(symbol)
        else:
            other_symbols.append(symbol)
    
    print(f'Equity symbols: {len(equity_symbols)}')
    print(f'Commodity symbols: {len(commodity_symbols)}')
    print(f'Other symbols: {len(other_symbols)}')
    print(f'Total: {len(rows)}')
    
    # Show first few symbols
    print(f'\nFirst 10 symbols: {[row.get("symbol") for row in rows[:10]]}')
    
    # Show all symbols if count is small
    if len(rows) <= 60:
        print(f'\nAll symbols: {[row.get("symbol") for row in rows]}')
    
except Exception as e:
    print(f'Error: {e}')
