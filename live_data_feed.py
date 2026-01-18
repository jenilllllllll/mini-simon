"""
Live Data Feed Module
Handles WebSocket connection to Fyers API and real-time candle management
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Callable, Optional
from collections import defaultdict, deque
import threading
import time as time_module
from fyers_apiv3 import fyersModel

logger = logging.getLogger(__name__)

class LiveCandleManager:
    """Manages real-time candle data for multiple symbols and timeframes"""
    
    def __init__(self, symbols: List[str], timeframes: List[str]):
        self.symbols = symbols
        self.timeframes = timeframes
        self.candles = defaultdict(lambda: defaultdict(deque))
        self.last_prices = {}
        self.callbacks = []
        self.max_candles = 500  # Keep last 500 candles per symbol/timeframe
        
        # Timeframe mappings in seconds
        self.tf_seconds = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '60m': 3600, '120m': 7200, '180m': 10800, '240m': 14400, '1D': 86400
        }
        
        # Indian market hours
        self.market_open = time(9, 15)
        self.market_close = time(15, 30)
        
    def add_callback(self, callback: Callable):
        """Add callback function for new candle events"""
        self.callbacks.append(callback)
        
    def _is_market_hours(self, dt: datetime = None) -> bool:
        """Check if current time is within market hours"""
        if dt is None:
            dt = datetime.now()
        current_time = dt.time()
        weekday = dt.weekday()
        
        # Check if it's a weekday (Monday=0 to Friday=4)
        if weekday > 4:
            return False
            
        # Check market hours (9:15 to 15:30)
        return self.market_open <= current_time <= self.market_close
        
    def _get_candle_timestamp(self, timestamp: int, timeframe: str) -> datetime:
        """Get candle start timestamp based on timeframe"""
        dt = datetime.fromtimestamp(timestamp)
        tf_secs = self.tf_seconds[timeframe]
        
        # Round down to nearest timeframe boundary
        if timeframe == '1D':
            # For daily, use 9:15 AM as start
            return dt.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # For intraday, round down to timeframe boundary
            seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            candle_start_seconds = int(seconds_since_midnight / tf_secs) * tf_secs
            candle_start = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=candle_start_seconds)
            return candle_start
            
    def update_tick(self, symbol: str, tick_data: dict):
        """Process incoming tick data and update candles"""
        if not self._is_market_hours():
            return
            
        try:
            timestamp = tick_data.get('timestamp', time_module.time())
            price = float(tick_data.get('price', 0))
            volume = int(tick_data.get('volume', 0))
            
            # Update last price
            self.last_prices[symbol] = price
            
            # Update candles for all timeframes
            for tf in self.timeframes:
                candle_time = self._get_candle_timestamp(timestamp, tf)
                self._update_candle(symbol, tf, candle_time, price, volume)
                
        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}")
            
    def _update_candle(self, symbol: str, timeframe: str, candle_time: datetime, price: float, volume: int):
        """Update candle for specific symbol and timeframe"""
        candles = self.candles[symbol][timeframe]
        
        # Get or create candle
        if candles and candles[-1]['timestamp'] == candle_time:
            # Update existing candle
            candle = candles[-1]
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] += volume
        else:
            # New candle
            candle = {
                'timestamp': candle_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': volume
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

class FyersWebSocketFeed:
    """Fyers WebSocket connection for real-time data"""
    
    def __init__(self, app_id: str, access_token: str, symbols: List[str], timeframes: List[str]):
        self.app_id = app_id
        self.access_token = access_token
        self.symbols = symbols
        self.timeframes = timeframes
        self.candle_manager = LiveCandleManager(symbols, timeframes)
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        
    def on_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'sf':
                # Tick data
                symbol = data.get('symbol')
                if symbol in self.symbols:
                    tick_data = {
                        'timestamp': data.get('timestamp', time_module.time()),
                        'price': data.get('ltp', 0),
                        'volume': data.get('volume', 0)
                    }
                    self.candle_manager.update_tick(symbol, tick_data)
                    
            elif data.get('type') == 'cf':
                # Candle data (if we subscribe to candle feed)
                pass
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self._attempt_reconnect()
        
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        self.reconnect_attempts = 0
        self._subscribe_symbols()
        
    def _subscribe_symbols(self):
        """Subscribe to symbols for real-time data"""
        try:
            # Subscribe to tick data for all symbols
            symbols_data = [{"symbol": symbol, "type": "sf"} for symbol in self.symbols]
            
            subscribe_message = {
                "type": "sf",
                "symbols": symbols_data
            }
            
            self.ws.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            
    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
            time_module.sleep(self.reconnect_delay)
            self.connect()
        else:
            logger.error("Max reconnection attempts reached")
            
    def connect(self):
        """Connect to Fyers WebSocket"""
        try:
            import websocket
            
            ws_url = f"wss://ws.fyers.in/v1/data-feed?access_token={self.access_token}&client_id={self.app_id}"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Start WebSocket in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
            # Wait for connection
            time_module.sleep(2)
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.is_connected = False
            
    def disconnect(self):
        """Disconnect WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False

class HistoricalDataBackfill:
    """Handles historical data backfill for strategy initialization"""
    
    def __init__(self, fyers_client):
        self.fyers_client = fyers_client
        
    def get_historical_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get historical candles for backfill"""
        try:
            # Skip historical data for now - use live data only
            logger.warning(f"Skipping historical data for {symbol} - using live data only")
            return pd.DataFrame()
            
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
                "symbol": symbol,
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

class LiveDataFeed:
    """Main live data feed orchestrator"""
    
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
        
        # Initialize components
        self.ws_feed = FyersWebSocketFeed(
            config['app_id'],
            config['access_token'],
            self.symbols,
            self.timeframes
        )
        
        self.backfill = HistoricalDataBackfill(self.fyers_client)
        
        # Callbacks for new candle events
        self.candle_callbacks = []
        
        # Register candle callback
        self.ws_feed.candle_manager.add_callback(self._on_new_candle)
        
    def _on_new_candle(self, symbol: str, timeframe: str, candle: dict):
        """Handle new candle formation"""
        for callback in self.candle_callbacks:
            try:
                callback(symbol, timeframe, candle)
            except Exception as e:
                logger.error(f"Error in candle callback: {e}")
                
    def add_candle_callback(self, callback: Callable):
        """Add callback for new candle events"""
        self.candle_callbacks.append(callback)
        
    def initialize(self):
        """Initialize data feed with historical backfill"""
        logger.info("Initializing live data feed...")
        
        # Backfill historical data for all symbols and timeframes
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                df = self.backfill.get_historical_candles(symbol, timeframe, 200)
                if not df.empty:
                    # Load into candle manager
                    for _, row in df.iterrows():
                        candle_data = {
                            'timestamp': row['date'],
                            'price': row['close'],
                            'volume': row['volume']
                        }
                        self.ws_feed.candle_manager.update_tick(symbol, candle_data)
                    logger.info(f"Backfilled {len(df)} candles for {symbol} {timeframe}")
                    
    def start(self):
        """Start the live data feed"""
        self.initialize()
        self.ws_feed.connect()
        logger.info("Live data feed started")
        
    def stop(self):
        """Stop the live data feed"""
        self.ws_feed.disconnect()
        logger.info("Live data feed stopped")
        
    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get recent candles"""
        return self.ws_feed.candle_manager.get_candles(symbol, timeframe, count)
        
    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get latest candle"""
        return self.ws_feed.candle_manager.get_latest_candle(symbol, timeframe)
        
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.ws_feed.is_connected
