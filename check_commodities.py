import requests

print('=== CHECKING COMMODITY WATCHLIST ===')

# Check main market feed
try:
    response = requests.get('http://127.0.0.1:8000/api/market-feed')
    data = response.json()
    main_rows = data.get('rows', [])
    print(f'Main market feed: {len(main_rows)} rows')
    
    # Check commodities feed
    response2 = requests.get('http://127.0.0.1:8000/api/market-feed-commodities')
    data2 = response2.json()
    commodity_rows = data2.get('rows', [])
    print(f'Commodity market feed: {len(commodity_rows)} rows')
    
    if commodity_rows:
        print(f'Commodity symbols: {[row.get("symbol") for row in commodity_rows[:5]]}')
    else:
        print('No commodities found')
        
except Exception as e:
    print(f'Error: {e}')
