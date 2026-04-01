import requests
import time

print('Testing commodities API call...')
time.sleep(3)

try:
    response = requests.get('http://127.0.0.1:8000/api/market-feed-commodities')
    data = response.json()
    rows = data.get('rows', [])
    print(f'API Response: {len(rows)} rows')
    print(f'Status: {response.status_code}')
    
    if rows:
        print(f'Commodities: {[row.get("symbol") for row in rows]}')
    else:
        print('❌ Still empty!')
        
except Exception as e:
    print(f'Error: {e}')
