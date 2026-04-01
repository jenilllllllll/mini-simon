import requests

print('=== COMMODITY WATCHLIST CHECK ===')

# Check commodities API
try:
    response = requests.get('http://127.0.0.1:8000/api/market-feed-commodities')
    data = response.json()
    rows = data.get('rows', [])
    print(f'Commodities API response: {len(rows)} rows')
    
    if rows:
        print(f'Commodities: {[row.get("symbol") for row in rows[:10]]}')
    else:
        print('❌ COMMODITY WATCHLIST IS EMPTY!')
        
except Exception as e:
    print(f'Error: {e}')

print('\n=== REQUIRED COMMODITIES ===')
required = ['GOLD', 'SILVER', 'CRUDE']
print(f'Must have: {required}')
