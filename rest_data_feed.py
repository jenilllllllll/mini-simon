"""
REST API Data Feed Module
Fallback solution using Fyers REST API polling when WebSocket is unavailable
"""

import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from collections import defaultdict, deque
import threading
from fyers_apiv3 import fyersModel

logger = logging.getLogger(__name__)

class RestDataFeed:
    """Data feed using Fyers REST API polling as WebSocket fallback"""
    
    def __init__(self, config):
        self.config = config
        self.symbols = config['symbols']
        self.timeframes = config['timeframes']
        
        # Initialize Fyers client
        self.fyers_client = fyersModel.FyersModel(
            client_id=config['app_id'],
            token=config['access_token'],
            log_path=config.get('log_path', '')
        )
        
        # Data storage
        self.candles = defaultdict(lambda: defaultdict(deque))
        self.callbacks = []
        self.max_candles = 1000
        
        # Polling control
        self.is_running = False
        self.poll_thread = None
        self.poll_interval = 30  # seconds
        
        # Cache for latest prices
        self.latest_prices = {}
        
    def add_callback(self, callback: Callable):
        """Add callback for new data"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: Callable):
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
        
    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get recent candles as DataFrame"""
        candles = list(self.candles[symbol][timeframe])[-count:]
        if not candles:
            return pd.DataFrame()
            
        df = pd.DataFrame(candles)
        df['date'] = df['timestamp']
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
        
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get latest candle for symbol and timeframe"""
        candles = self.candles[symbol][timeframe]
        return candles[-1] if candles else None
        
    def start(self):
        """Start the polling thread"""
        if not self.is_running:
            self.is_running = True
            self.poll_thread = threading.Thread(target=self._poll_loop)
            self.poll_thread.daemon = True
            self.poll_thread.start()
            logger.info("REST data feed started")
            
    def stop(self):
        """Stop the polling thread"""
        self.is_running = False
        if self.poll_thread:
            self.poll_thread.join(timeout=5)
        logger.info("REST data feed stopped")
        
    def _poll_loop(self):
        """Main polling loop"""
        logger.info("Starting REST API polling loop")
        
        while self.is_running:
            try:
                self._fetch_all_data()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                time.sleep(5)  # Short delay on error
                
    def _fetch_all_data(self):
        """Fetch data for all symbols and timeframes"""
        # Fetch latest quotes for all symbols
        try:
            symbols_str = ','.join([f"NSE:{symbol}" for symbol in self.symbols])
            quotes_response = self.fyers_client.quotes({'symbols': symbols_str})
            
            if quotes_response.get('s') == 'ok':
                for quote_data in quotes_response.get('d', []):
                    if quote_data.get('v', {}).get('s') == 'ok':
                        symbol = quote_data['n'].replace('NSE:', '')
                        price_data = quote_data['v']
                        
                        # Update latest price
                        self.latest_prices[symbol] = price_data.get('lp', 0)
                        
                        # Create tick data
                        tick_data = {
                            'timestamp': datetime.now(),
                            'price': price_data.get('lp', 0),
                            'volume': price_data.get('tv', 0)
                        }
                        
                        # Update candles for all timeframes
                        for timeframe in self.timeframes:
                            self._update_candle(symbol, timeframe, tick_data)
                            
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            
    def _update_candle(self, symbol: str, timeframe: str, tick_data: dict):
        """Update candle for specific symbol and timeframe"""
        candles = self.candles[symbol][timeframe]
        
        # Calculate candle time based on timeframe
        now = tick_data['timestamp']
        if timeframe == '1D':
            candle_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            minutes = int(timeframe.replace('m', ''))
            candle_time = now.replace(second=0, microsecond=0)
            candle_time = candle_time.replace(
                minute=(candle_time.minute // minutes) * minutes
            )
        
        # Get or create candle
        if candles and candles[-1]['timestamp'] == candle_time:
            # Update existing candle
            candle = candles[-1]
            price = tick_data['price']
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += tick_data['volume']
        else:
            # New candle
            price = tick_data['price']
            candle = {
                'timestamp': candle_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': tick_data['volume']
            }
            candles.append(candle)
            
            # Trigger callbacks for new candle
            for callback in self.callbacks:
                try:
                    callback(symbol, timeframe, candle)
                except Exception as e:
                    logger.error(f"Error in candle callback: {e}")
                    
        # Limit candle history
        if len(candles) > self.max_candles:
            candles.popleft()
            
    def get_historical_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get historical candles using REST API"""
        try:
            # Convert timeframe to Fyers format
            tf_mapping = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15',
                '60m': '60', '120m': '120', '180m': '180', '240m': '240', '1D': 'D'
            }
            
            fyers_tf = tf_mapping.get(timeframe, timeframe)
            
            # Calculate start date
            end_date = datetime.now()
            if timeframe == '1D':
                start_date = end_date - timedelta(days=count)
            else:
                # Approximate minutes needed
                tf_minutes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '60m': 60, '120m': 120, '180m': 180, '240m': 240}
                minutes_needed = count * tf_minutes.get(timeframe, 60)
                start_date = end_date - timedelta(minutes=minutes_needed)
                
            # Get historical data
            data = {
                "symbol": f"NSE:{symbol}",
                "resolution": fyers_tf,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": end_date.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            response = self.fyers_client.history(data)
            
            if response.get('s') == 'ok':
                candles = response['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                return df.tail(count)  # Return only requested count
            else:
                logger.error(f"Error getting historical data for {symbol}: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting historical candles for {symbol}: {e}")
            return pd.DataFrame()
