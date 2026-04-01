# Current NIFTY-50 in system (49 symbols)
current_symbols = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'LT', 'ITC', 'SBIN', 'HINDUNILVR', 
    'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'WIPRO', 'ULTRACEMCO', 
    'NESTLEIND', 'POWERGRID', 'BAJAJFINSV', 'TECHM', 'NTPC', 'GRASIM', 'JSWSTEEL', 'HCLTECH', 
    'TATAMOTORS', 'DRREDDY', 'CIPLA', 'ONGC', 'HDFCLIFE', 'DIVISLAB', 'HEROMOTOCO', 'BRITANNIA', 
    'BPCL', 'COALINDIA', 'ADANIENT', 'ADANIPORTS', 'INDUSINDBK', 'BAJAJ-AUTO', 'EICHERMOT', 
    'TATACONSUM', 'HINDALCO', 'APOLLOHOSP', 'TATASTEEL', 'M&M', 'BHARTIARTL', 'SHRIRAMFIN', 'JIOFINANCE'
]

# Official NIFTY-50 list (50 symbols)
official_nifty_50 = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'LT', 'ITC', 'SBIN', 'HINDUNILVR',
    'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'WIPRO', 'ULTRACEMCO',
    'NESTLEIND', 'POWERGRID', 'BAJAJFINSV', 'TECHM', 'NTPC', 'GRASIM', 'JSWSTEEL', 'HCLTECH',
    'TATAMOTORS', 'DRREDDY', 'CIPLA', 'ONGC', 'HDFCLIFE', 'DIVISLAB', 'HEROMOTOCO', 'BRITANNIA',
    'BPCL', 'COALINDIA', 'ADANIENT', 'ADANIPORTS', 'INDUSINDBK', 'BAJAJ-AUTO', 'EICHERMOT',
    'TATACONSUM', 'HINDALCO', 'APOLLOHOSP', 'TATASTEEL', 'M&M', 'BHARTIARTL', 'SHRIRAMFIN', 'JIOFINANCE',
    'UPL'  # This is the missing one!
]

# Find missing symbol
missing = set(official_nifty_50) - set(current_symbols)
print(f'Missing NIFTY-50 symbol: {missing}')
print(f'Current count: {len(current_symbols)}')
print(f'Should be: {len(official_nifty_50)}')
