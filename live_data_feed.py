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

import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

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

        # MCX commodity market hours (rough)
        self.mcx_market_open = time(9, 0)
        self.mcx_market_close = time(23, 30)
        
    def add_callback(self, callback: Callable):
        """Add callback function for new candle events"""
        self.callbacks.append(callback)

    def _is_market_open_for_symbol(self, symbol: str, dt: Optional[datetime] = None) -> bool:
        if dt is None:
            dt = datetime.now(IST)

        weekday = dt.weekday()
        if weekday > 4:
            return False

        current_time = dt.time()
        sym = str(symbol or "").upper()
        if sym.startswith("MCX:"):
            return self.mcx_market_open <= current_time <= self.mcx_market_close

        return self.market_open <= current_time <= self.market_close
        
    def _is_market_hours(self, dt: datetime = None) -> bool:
        """Check if current time is within market hours"""
        if dt is None:
            dt = datetime.now(IST)
        current_time = dt.time()
        weekday = dt.weekday()
        
        # Check if it's a weekday (Monday=0 to Friday=4)
        if weekday > 4:
            return False
            
        # Check market hours (9:15 to 15:30)
        return self.market_open <= current_time <= self.market_close
        
    def _get_candle_timestamp(self, timestamp: int, timeframe: str) -> datetime:
        """Get candle start timestamp based on timeframe"""
        dt = datetime.fromtimestamp(timestamp, IST)
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
        if not self._is_market_open_for_symbol(symbol):
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
            
    def _update_candle(
        self,
        symbol: str,
        timeframe: str,
        candle_time: datetime,
        price: float,
        volume: int,
        trigger_callbacks: bool = True,
    ):
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
            
            if trigger_callbacks:
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
    """Fyers WebSocket connection for real-time data with keep-alive heartbeat."""
    
    def __init__(self, app_id: str, access_token: str, symbols: List[str], timeframes: List[str]):
        self.app_id = app_id
        self.access_token = access_token
        self.symbols = symbols
        self.timeframes = timeframes
        self.candle_manager = LiveCandleManager(symbols, timeframes)
        self.ws = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10  # Increased for cloud stability
        self.reconnect_delay = 5
        
        # Keep-alive heartbeat configuration
        self.heartbeat_interval = 30  # seconds
        self.last_message_time = 0
        self.heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        
        # Connection monitoring
        self.last_tick_time = 0
        self.tick_timeout = 90  # Reconnect if no tick for 90 seconds
        self._connection_monitor_thread = None
        
    def on_message(self, ws, message):
        """Handle WebSocket messages"""
        try:
            self.last_message_time = time_module.time()
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
                    self.last_tick_time = time_module.time()
                    
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
        self._stop_heartbeat.set()
        self._attempt_reconnect()
        
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        self.reconnect_attempts = 0
        self.last_message_time = time_module.time()
        self.last_tick_time = time_module.time()
        self._stop_heartbeat.clear()
        self._subscribe_symbols()
        self._start_heartbeat()
        self._start_connection_monitor()
        
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
            
    def _start_heartbeat(self):
        """Start keep-alive heartbeat thread"""
        def heartbeat_loop():
            while not self._stop_heartbeat.is_set():
                try:
                    time_module.sleep(self.heartbeat_interval)
                    if self.is_connected and self.ws:
                        # Send heartbeat/ping to keep connection alive
                        try:
                            # Some WebSocket implementations support ping
                            self.ws.sock.ping()
                            logger.debug("WebSocket heartbeat ping sent")
                        except AttributeError:
                            # Fallback: just log that we're still connected
                            pass
                except Exception as e:
                    logger.debug(f"Heartbeat error (expected during shutdown): {e}")
                    
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        logger.info("WebSocket heartbeat started")
        
    def _start_connection_monitor(self):
        """Monitor connection health and reconnect if needed"""
        def monitor_loop():
            while not self._stop_heartbeat.is_set():
                try:
                    time_module.sleep(10)  # Check every 10 seconds
                    if self.is_connected:
                        now = time_module.time()
                        # Check if we've received any message recently
                        if self.last_message_time > 0 and (now - self.last_message_time) > self.tick_timeout:
                            logger.warning(f"No message received for {int(now - self.last_message_time)}s, forcing reconnect")
                            self._force_reconnect()
                except Exception as e:
                    logger.error(f"Connection monitor error: {e}")
                    
        self._connection_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._connection_monitor_thread.start()
        logger.info("Connection monitor started")
        
    def _force_reconnect(self):
        """Force a reconnection of the WebSocket"""
        try:
            if self.ws:
                self.ws.close()
            self.is_connected = False
            self._stop_heartbeat.set()
            time_module.sleep(1)
            self._attempt_reconnect()
        except Exception as e:
            logger.error(f"Error during forced reconnect: {e}")
            
    def _attempt_reconnect(self):
        """Attempt to reconnect WebSocket with exponential backoff"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            # Exponential backoff: delay increases with each attempt
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
            logger.info(f"Attempting to reconnect ({self.reconnect_attempts}/{self.max_reconnect_attempts}) after {delay}s")
            time_module.sleep(delay)
            self.connect()
        else:
            logger.error("Max reconnection attempts reached. Will retry in 5 minutes.")
            # Reset after 5 minutes for cloud stability
            time_module.sleep(300)
            self.reconnect_attempts = 0
            self._attempt_reconnect()
            
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
        """Disconnect WebSocket cleanly"""
        self._stop_heartbeat.set()
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("WebSocket disconnected")

class HistoricalDataBackfill:
    """Handles historical data backfill for strategy initialization"""
    
    def __init__(self, fyers_client):
        self.fyers_client = fyers_client
        
    def get_historical_candles(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get historical candles for backfill"""
        try:
            # Skip historical data for now - use live data only
            logger.info(f"Requesting historical data for {symbol} {timeframe}, count={count}")
            
            # Convert timeframe to Fyers format
            tf_mapping = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15',
                '60m': '60', '120m': '120', '180m': '180', '240m': '240', '1D': 'D'
            }
            
            fyers_tf = tf_mapping.get(timeframe, timeframe)
            
            # Calculate start date
            end_date = datetime.now(IST)
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
                df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(IST)
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

        token_raw = str(config.get('access_token') or '').strip()
        if ':' in token_raw:
            token_raw = token_raw.split(':', 1)[1]
        
        # Initialize Fyers client
        self.fyers_client = fyersModel.FyersModel(
            client_id=config['app_id'],
            token=token_raw,
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
        logger.info("Initializing live data feed with historical backfill...")

        for i, symbol in enumerate(self.symbols):
            for timeframe in self.timeframes:
                # Add delay to stay within Fyers API rate limits (10/sec)
                # 60 symbols * 3 timeframes = 180 requests.
                if i > 0:
                    time_module.sleep(0.3)
                
                # Fetch with retry logic
                df = pd.DataFrame()
                for attempt in range(2):
                    df = self.backfill.get_historical_candles(symbol, timeframe, 200)
                    if not df.empty:
                        break
                    
                    # If we hit rate limit or error, wait and retry once
                    logger.warning(f"Backfill missing for {symbol} {timeframe}. Attempt {attempt+1} failed.")
                    time_module.sleep(1.0)

                if not df.empty:
                    # Load into candle manager
                    for _, row in df.iterrows():
                        candle_time = row["date"]
                        if hasattr(candle_time, "to_pydatetime"):
                            candle_time = candle_time.to_pydatetime()
                        price = float(row["close"])
                        volume = int(row["volume"])
                        self.ws_feed.candle_manager._update_candle(
                            symbol,
                            timeframe,
                            candle_time,
                            price,
                            volume,
                            trigger_callbacks=False,
                        )
                    logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
                else:
                    logger.error(f"❌ Failed to backfill {symbol} {timeframe}")

                    
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
