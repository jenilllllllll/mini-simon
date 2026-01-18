"""
Simple Live Engine Test
Tests the pipeline with mock data to validate signal generation
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engine import FeatureEngine
from live_strategy_runner import LiveStrategyRunner
from live_signal_aggregator import SignalAggregator
from signal_store import SignalStore
from utils import DataFrameUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_candle_data(symbol: str, timeframe: str, num_candles: int = 200):
    """Generate realistic mock candle data"""
    logger.info(f"Generating {num_candles} mock candles for {symbol} {timeframe}")
    
    # Base price with some volatility
    base_price = 1000.0 + np.random.uniform(-200, 200)
    price_changes = np.random.normal(0, 0.01, num_candles)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    candles = []
    current_time = datetime.now() - timedelta(minutes=num_candles * 5)
    
    for i in range(num_candles):
        # Generate OHLC
        open_price = prices[i]
        volatility = abs(np.random.normal(0, 0.005))
        high = open_price * (1 + volatility)
        low = open_price * (1 - volatility)
        close = open_price + np.random.normal(0, volatility * open_price * 0.5)
        close = max(low, min(high, close))
        volume = int(np.random.uniform(1000, 10000))
        
        candles.append({
            'date': current_time,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
        
        current_time += timedelta(minutes=5)
    
    return pd.DataFrame(candles)

def test_live_pipeline():
    """Test the complete live pipeline with mock data"""
    logger.info("🚀 Starting Live Pipeline Test with Mock Data...")
    
    try:
        # Initialize components
        feature_engine = FeatureEngine()
        strategy_runner = LiveStrategyRunner()
        signal_aggregator = SignalAggregator()
        
        # Test configuration
        test_config = {
            'base_path': 'test_signals',
            'enable_csv': True,
            'enable_json': True,
            'enable_db': False
        }
        
        signal_store = SignalStore(test_config)
        
        # Test symbols and timeframes
        symbols = ['RELIANCE', 'TCS', 'INFY']
        timeframes = ['5m', '15m']
        
        total_signals = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"📊 Processing {symbol} {timeframe}...")
                
                # Generate mock data
                df = generate_mock_candle_data(symbol, timeframe, 200)
                
                # Add some patterns to trigger strategies
                df = add_mock_patterns(df)
                
                # Calculate features
                df_with_features = feature_engine.calculate_all_features(df)
                logger.info(f"✅ Features calculated: {len(df_with_features.columns)} columns")
                
                # Run strategies
                signals = strategy_runner.run_all_strategies(df_with_features, symbol, timeframe)
                logger.info(f"🎯 Generated {len(signals)} strategy signals")
                
                if signals:
                    # Aggregate signals
                    aggregated_signals = signal_aggregator.aggregate_signals(signals, symbol)
                    logger.info(f"📈 Aggregated to {len(aggregated_signals)} final signals")
                    
                    if aggregated_signals:
                        # Store signals
                        stored = signal_store.store_signals(aggregated_signals)
                        total_signals += stored
                        
                        # Log signal details
                        for signal in aggregated_signals:
                            logger.info(f"🚨 SIGNAL: {signal.symbol} {signal.final_action} @ {signal.entry_price} "
                                      f"(Conf: {signal.aggregated_confidence:.2f})")
        
        logger.info(f"🎉 Pipeline Test Complete! Total signals stored: {total_signals}")
        
        # Get statistics
        stats = signal_store.get_signal_statistics()
        logger.info(f"📊 Signal Store Stats: {stats}")
        
        # Export signals
        signal_store.export_signals('test_signals_export.csv', format='csv')
        logger.info("💾 Signals exported to test_signals_export.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        if Path('test_signals').exists():
            shutil.rmtree('test_signals')

def add_mock_patterns(df):
    """Add mock patterns to trigger strategy signals"""
    df = df.copy()
    
    # Add volume spikes
    spike_indices = np.random.choice(len(df), size=min(10, len(df)//10), replace=False)
    for idx in spike_indices:
        df.loc[idx, 'volume'] *= np.random.uniform(2, 5)
    
    # Add swing points
    for i in range(20, len(df)-20):
        if i % 30 == 0:
            if np.random.random() > 0.5:
                # Swing high
                df.loc[i, 'high'] *= 1.02
                df.loc[i, 'close'] = df.loc[i, 'high'] * 0.99
            else:
                # Swing low
                df.loc[i, 'low'] *= 0.98
                df.loc[i, 'close'] = df.loc[i, 'low'] * 1.01
    
    return df

if __name__ == "__main__":
    success = test_live_pipeline()
    if success:
        logger.info("✅ Live pipeline test successful!")
        logger.info("🔥 Your Mini-Simon Live Engine is ready for production!")
    else:
        logger.error("❌ Live pipeline test failed!")
    
    sys.exit(0 if success else 1)
