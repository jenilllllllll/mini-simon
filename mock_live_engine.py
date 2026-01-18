"""
Mock Live Engine
Simulates live data with realistic market patterns for testing
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from logger_config import LoggerConfig
from feature_engine import FeatureEngine
from live_strategy_runner import LiveStrategyRunner
from live_signal_aggregator import SignalAggregator
from signal_store import SignalStore

class MockDataFeed:
    """Mock data feed that generates realistic market data"""
    
    def __init__(self, symbols, timeframes):
        self.symbols = symbols
        self.timeframes = timeframes
        self.is_running = False
        self.callbacks = []
        self.candles = {}
        
        # Initialize candle storage
        for symbol in symbols:
            self.candles[symbol] = {}
            for timeframe in timeframes:
                self.candles[symbol][timeframe] = []
        
        # Generate initial historical data
        self._generate_initial_data()
        
    def add_callback(self, callback):
        """Add callback for new candle"""
        self.callbacks.append(callback)
        
    def start(self):
        """Start the mock data feed"""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_simulation)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the mock data feed"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def get_candles(self, symbol, timeframe, count=100):
        """Get candles for symbol and timeframe"""
        candles = self.candles.get(symbol, {}).get(timeframe, [])
        df = pd.DataFrame(candles[-count:])
        if not df.empty:
            df['date'] = df['timestamp']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
        
    def _generate_initial_data(self):
        """Generate initial historical data"""
        for symbol in self.symbols:
            base_price = 1000 + random.uniform(-500, 500)
            
            for timeframe in self.timeframes:
                candles = []
                current_time = datetime.now() - timedelta(hours=24)
                
                # Generate 200 initial candles
                for i in range(200):
                    # Generate realistic OHLC
                    price_change = np.random.normal(0, 0.01)
                    open_price = base_price * (1 + price_change)
                    
                    volatility = abs(np.random.normal(0, 0.02))
                    high = open_price * (1 + volatility)
                    low = open_price * (1 - volatility)
                    close = open_price + np.random.normal(0, volatility * open_price * 0.3)
                    close = max(low, min(high, close))
                    volume = int(np.random.uniform(1000, 50000))
                    
                    candle = {
                        'timestamp': current_time,
                        'open': round(open_price, 2),
                        'high': round(high, 2),
                        'low': round(low, 2),
                        'close': round(close, 2),
                        'volume': volume
                    }
                    
                    candles.append(candle)
                    base_price = close
                    current_time += self._get_timeframe_delta(timeframe)
                
                self.candles[symbol][timeframe] = candles
                
    def _get_timeframe_delta(self, timeframe):
        """Get timedelta for timeframe"""
        if timeframe == '1m':
            return timedelta(minutes=1)
        elif timeframe == '3m':
            return timedelta(minutes=3)
        elif timeframe == '5m':
            return timedelta(minutes=5)
        elif timeframe == '15m':
            return timedelta(minutes=15)
        elif timeframe == '60m':
            return timedelta(hours=1)
        elif timeframe == '120m':
            return timedelta(hours=2)
        elif timeframe == '180m':
            return timedelta(hours=3)
        elif timeframe == '240m':
            return timedelta(hours=4)
        else:
            return timedelta(minutes=5)
            
    def _run_simulation(self):
        """Run the market simulation"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        # Generate new candle if timeframe matches
                        if self._should_generate_candle(timeframe, current_time):
                            candle = self._generate_new_candle(symbol, timeframe, current_time)
                            self.candles[symbol][timeframe].append(candle)
                            
                            # Keep only last 1000 candles
                            if len(self.candles[symbol][timeframe]) > 1000:
                                self.candles[symbol][timeframe].pop(0)
                            
                            # Trigger callbacks
                            for callback in self.callbacks:
                                try:
                                    callback(symbol, timeframe, candle)
                                except Exception as e:
                                    logging.error(f"Error in callback: {e}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logging.error(f"Error in simulation: {e}")
                time.sleep(5)
                
    def _should_generate_candle(self, timeframe, current_time):
        """Check if we should generate a new candle for this timeframe"""
        candles = self.candles.get('RELIANCE', {}).get(timeframe, [])
        if not candles:
            return True
            
        last_candle_time = candles[-1]['timestamp']
        delta = self._get_timeframe_delta(timeframe)
        
        return current_time - last_candle_time >= delta
        
    def _generate_new_candle(self, symbol, timeframe, current_time):
        """Generate a new candle"""
        candles = self.candles[symbol][timeframe]
        last_close = candles[-1]['close'] if candles else 1000
        
        # Add some patterns to trigger strategies
        if random.random() < 0.1:  # 10% chance of pattern
            if random.random() < 0.5:
                # Volume spike
                volume_multiplier = random.uniform(2, 5)
            else:
                # Large price move
                price_change = np.random.normal(0, 0.03)
                volume_multiplier = 1.5
        else:
            price_change = np.random.normal(0, 0.01)
            volume_multiplier = 1.0
        
        open_price = last_close * (1 + price_change)
        volatility = abs(np.random.normal(0, 0.015))
        high = open_price * (1 + volatility)
        low = open_price * (1 - volatility)
        close = open_price + np.random.normal(0, volatility * open_price * 0.3)
        close = max(low, min(high, close))
        volume = int(np.random.uniform(1000, 20000) * volume_multiplier)
        
        return {
            'timestamp': current_time,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        }

class MockLiveEngine:
    """Mock live engine for testing"""
    
    def __init__(self, config):
        self.config = config
        self.is_running = False
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.strategy_runner = LiveStrategyRunner()
        self.signal_aggregator = SignalAggregator()
        self.signal_store = SignalStore(config.get('signal_store', {}))
        
        # Initialize mock data feed
        symbols = config.get('data_feed', {}).get('symbols', ['RELIANCE', 'TCS', 'INFY'])
        timeframes = config.get('data_feed', {}).get('timeframes', ['5m', '15m'])
        self.data_feed = MockDataFeed(symbols, timeframes)
        self.data_feed.add_callback(self._on_new_candle)
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'signals_stored': 0,
            'errors': 0,
            'start_time': None
        }
        
    def start(self):
        """Start the mock engine"""
        try:
            logging.info("Starting Mock Live Engine...")
            
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # Start data feed
            self.data_feed.start()
            
            logging.info("Mock Live Engine started successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error starting mock engine: {e}")
            return False
            
    def stop(self):
        """Stop the mock engine"""
        logging.info("Stopping Mock Live Engine...")
        
        self.is_running = False
        self.data_feed.stop()
        
        logging.info(f"Final Stats - Signals Generated: {self.stats['signals_generated']}")
        logging.info(f"Final Stats - Signals Stored: {self.stats['signals_stored']}")
        logging.info("Mock Live Engine stopped")
        
    def _on_new_candle(self, symbol, timeframe, candle):
        """Handle new candle"""
        try:
            # Get recent data
            df = self.data_feed.get_candles(symbol, timeframe, count=200)
            
            if len(df) < 50:
                return
                
            # Calculate features
            df_with_features = self.feature_engine.calculate_all_features(df)
            
            # Run strategies
            signals = self.strategy_runner.run_all_strategies(df_with_features, symbol, timeframe)
            
            if signals:
                logging.info(f"Generated {len(signals)} signals for {symbol} {timeframe}")
                
                # Aggregate signals
                aggregated_signals = self.signal_aggregator.aggregate_signals(signals, symbol)
                
                if aggregated_signals:
                    # Store signals
                    stored_count = self.signal_store.store_signals(aggregated_signals)
                    self.stats['signals_generated'] += len(aggregated_signals)
                    self.stats['signals_stored'] += stored_count
                    
                    # Log signal details
                    for signal in aggregated_signals[:2]:  # Log first 2 signals
                        logging.info(f"SIGNAL: {signal.symbol} {signal.final_action} "
                                  f"@ {signal.entry_price} (Conf: {signal.aggregated_confidence:.2f})")
                        
        except Exception as e:
            logging.error(f"Error processing candle: {e}")
            self.stats['errors'] += 1

def main():
    """Main function"""
    try:
        # Setup logging
        LoggerConfig.setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Mock Live Engine...")
        
        # Load configuration
        config = Config()
        
        # Create mock engine
        engine = MockLiveEngine(config.config_data)
        
        # Start engine
        if not engine.start():
            logger.error("Engine start failed")
            return False
        
        # Run for specified duration or until interrupted
        try:
            duration = 300  # 5 minutes
            logger.info(f"Running mock engine for {duration} seconds...")
            
            for i in range(duration):
                if not engine.is_running:
                    break
                    
                if i % 30 == 0:  # Log every 30 seconds
                    logger.info(f"Running... Signals: {engine.stats['signals_generated']}, "
                              f"Errors: {engine.stats['errors']}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Stop engine
        engine.stop()
        
        # Export signals
        engine.signal_store.export_signals('mock_signals.csv', format='csv')
        logger.info("Signals exported to mock_signals.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
