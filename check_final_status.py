from web_main import NIFTY_50_SYMBOLS, COMMODITY_SYMBOLS

print('=== FINAL CONFIGURATION STATUS ===')
print(f'NIFTY-50 Stocks: {len(NIFTY_50_SYMBOLS)} symbols')
print(f'Commodity Symbols: {len(COMMODITY_SYMBOLS)} symbols')
print(f'Total Symbols in System: {len(NIFTY_50_SYMBOLS) + len(COMMODITY_SYMBOLS)}')

print(f'\nNIFTY-50 includes UPL: {"UPL" in NIFTY_50_SYMBOLS}')
print(f'Commodities configured: {len(COMMODITY_SYMBOLS) > 0}')

print(f'\nDashboard should show:')
print(f'- Main watchlist: 50 NIFTY-50 stocks only')
print(f'- Commodity watchlist: 3 commodities (GOLD, SILVER, CRUDE)')
print(f'- Total system coverage: 53 symbols')
