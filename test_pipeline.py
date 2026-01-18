"""
Test Pipeline Module
Validates the complete live trading pipeline with mock data
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engine import FeatureEngine
from live_strategy_runner import LiveStrategyRunner
from live_signal_aggregator import SignalAggregator, SignalConsolidator
from signal_store import SignalStore
from utils import DataFrameUtils, ValidationUtils

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generate realistic test data for pipeline validation"""
    
    @staticmethod
    def generate_ohlcv_data(symbol: str, days: int = 30, timeframe: str = '5m') -> pd.DataFrame:
        """Generate realistic OHLCV test data"""
        logger.info(f"Generating test data for {symbol} - {days} days, {timeframe} timeframe")
        
        # Calculate number of bars
        if timeframe == '1D':
            bars_per_day = 1
        else:
            minutes_map = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '60m': 60, '120m': 120, '180m': 180, '240m': 240}
            bars_per_day = (6 * 60) // minutes_map.get(timeframe, 5)  # 6 hours of trading
        
        total_bars = days * bars_per_day
        
        # Generate base price with trend and volatility
        base_price = 1000.0
        trend = np.random.normal(0.0001, 0.0002, total_bars)  # Small upward trend
        volatility = np.random.normal(0, 0.01, total_bars)  # 1% volatility
        
        # Generate price series
        price_changes = trend + volatility
        close_prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Generate OHLC data
        data = []
        current_time = datetime.now() - timedelta(days=days)
        
        for i in range(total_bars):
            close = close_prices[i]
            
            # Generate realistic OHLC relationships
            intrabar_volatility = abs(np.random.normal(0, 0.005))
            high = close * (1 + intrabar_volatility)
            low = close * (1 - intrabar_volatility)
            
            # Open price (previous close or with gap)
            if i == 0:
                open_price = close * (1 + np.random.normal(0, 0.001))
            else:
                gap = np.random.normal(0, 0.002)
                open_price = close_prices[i-1] * (1 + gap)
                open_price = max(low, min(high, open_price))  # Ensure open is within high-low
            
            # Generate volume
            base_volume = 10000
            volume_multiplier = 1 + abs(np.random.normal(0, 0.5))  # Random volume spikes
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'date': current_time,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
            
            # Increment time
            if timeframe == '1D':
                current_time += timedelta(days=1)
            else:
                minutes_map = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '60m': 60, '120m': 120, '180m': 180, '240m': 240}
                current_time += timedelta(minutes=minutes_map.get(timeframe, 5))
                
            # Skip weekends and non-trading hours
            if current_time.weekday() >= 5:  # Weekend
                current_time += timedelta(days=(7 - current_time.weekday()))
                current_time = current_time.replace(hour=9, minute=15)
            elif current_time.hour >= 15:  # After market close
                current_time = (current_time + timedelta(days=1)).replace(hour=9, minute=15)
                
        return pd.DataFrame(data)
    
    @staticmethod
    def add_pattern_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Add specific patterns to trigger strategy signals"""
        df = df.copy()
        
        # Add volume spike pattern
        spike_indices = np.random.choice(len(df), size=5, replace=False)
        for idx in spike_indices:
            df.loc[idx, 'volume'] *= 3.0  # Volume spike
            
        # Add swing points
        for i in range(10, len(df) - 10):
            if i % 50 == 0:  # Every 50 bars
                # Create swing high
                if np.random.random() > 0.5:
                    df.loc[i, 'high'] *= 1.02  # 2% higher
                    df.loc[i, 'close'] = df.loc[i, 'high'] * 0.99
                else:
                    # Create swing low
                    df.loc[i, 'low'] *= 0.98  # 2% lower
                    df.loc[i, 'close'] = df.loc[i, 'low'] * 1.01
                    
        return df

class PipelineTest:
    """Test the complete pipeline"""
    
    def __init__(self):
        self.test_results = {
            'feature_engine': False,
            'strategy_runner': False,
            'signal_aggregator': False,
            'signal_store': False,
            'end_to_end': False
        }
        
    def run_all_tests(self):
        """Run all pipeline tests"""
        logger.info("Starting pipeline validation tests...")
        
        # Test individual components
        self.test_feature_engine()
        self.test_strategy_runner()
        self.test_signal_aggregator()
        self.test_signal_store()
        
        # Test end-to-end pipeline
        self.test_end_to_end_pipeline()
        
        # Print results
        self.print_test_results()
        
        return all(self.test_results.values())
        
    def test_feature_engine(self):
        """Test feature engine"""
        try:
            logger.info("Testing Feature Engine...")
            
            # Generate test data
            df = TestDataGenerator.generate_ohlcv_data('TEST', 5, '5m')
            df = TestDataGenerator.add_pattern_signals(df)
            
            # Initialize feature engine
            feature_engine = FeatureEngine()
            
            # Calculate features
            df_with_features = feature_engine.calculate_all_features(df)
            
            # Validate results
            assert not df_with_features.empty, "Feature engine returned empty DataFrame"
            assert len(df_with_features) == len(df), "Feature engine changed row count"
            assert len(df_with_features.columns) > len(df.columns), "No features added"
            
            # Check for key features
            required_features = ['vwap', 'volume_spike', 'swing_high', 'swing_low', 'atr']
            for feature in required_features:
                assert feature in df_with_features.columns, f"Missing feature: {feature}"
                
            logger.info("✅ Feature Engine test passed")
            self.test_results['feature_engine'] = True
            
        except Exception as e:
            logger.error(f"❌ Feature Engine test failed: {e}")
            
    def test_strategy_runner(self):
        """Test strategy runner"""
        try:
            logger.info("Testing Strategy Runner...")
            
            # Generate test data
            df = TestDataGenerator.generate_ohlcv_data('TEST', 10, '5m')
            df = TestDataGenerator.add_pattern_signals(df)
            
            # Add basic features
            df = DataFrameUtils.add_basic_features(df)
            
            # Initialize strategy runner
            strategy_runner = LiveStrategyRunner()
            
            # Run strategies
            signals = strategy_runner.run_all_strategies(df, 'TEST', '5m')
            
            # Validate results
            assert isinstance(signals, list), "Strategy runner should return list"
            
            if signals:
                # Check signal structure
                signal = signals[0]
                required_fields = ['strategy_name', 'symbol', 'action', 'entry_price', 'stop_loss', 'target']
                for field in required_fields:
                    assert hasattr(signal, field), f"Missing signal field: {field}"
                    
            logger.info(f"✅ Strategy Runner test passed - Generated {len(signals)} signals")
            self.test_results['strategy_runner'] = True
            
        except Exception as e:
            logger.error(f"❌ Strategy Runner test failed: {e}")
            
    def test_signal_aggregator(self):
        """Test signal aggregator"""
        try:
            logger.info("Testing Signal Aggregator...")
            
            from live_strategy_runner import StrategySignal
            
            # Create test signals
            signals = [
                StrategySignal('vol_spike', 'TEST', '5m', 'BUY', 1000, 990, 1020, 0.7, 995),
                StrategySignal('body_imbalance', 'TEST', '15m', 'BUY', 1001, 991, 1021, 0.6, 996),
                StrategySignal('order_block', 'TEST', '5m', 'SELL', 1002, 1008, 992, 0.5, 1005)
            ]
            
            # Initialize aggregator
            aggregator = SignalAggregator()
            
            # Aggregate signals
            aggregated = aggregator.aggregate_signals(signals, 'TEST')
            
            # Validate results
            assert isinstance(aggregated, list), "Aggregator should return list"
            
            if aggregated:
                agg_signal = aggregated[0]
                required_fields = ['symbol', 'final_action', 'aggregated_confidence', 'decision_score']
                for field in required_fields:
                    assert hasattr(agg_signal, field), f"Missing aggregated signal field: {field}"
                    
            logger.info(f"✅ Signal Aggregator test passed - Aggregated to {len(aggregated)} signals")
            self.test_results['signal_aggregator'] = True
            
        except Exception as e:
            logger.error(f"❌ Signal Aggregator test failed: {e}")
            
    def test_signal_store(self):
        """Test signal store"""
        try:
            logger.info("Testing Signal Store...")
            
            from live_signal_aggregator import AggregatedSignal
            
            # Create test signal
            test_signal = AggregatedSignal(
                symbol='TEST',
                action='BUY',
                confidence=0.7,
                entry_price=1000.0,
                stop_loss_level=990.0,
                target_level=1020.0,
                signal_timestamp=datetime.now(),
                signal_timeframe='5m',
                chart_anchor_price=995.0,
                timestamp_generated=datetime.now(),
                aggregated_confidence=0.7,
                final_action='BUY',
                contributing_strategies=['vol_spike', 'body_imbalance'],
                contributing_timeframes=['5m', '15m'],
                decision_score=0.65,
                version_id='test_v1.0.0',
                raw_strategy_outputs=[],
                metadata={'test': True}
            )
            
            # Initialize signal store with test config
            test_config = {
                'base_path': 'test_signals',
                'enable_csv': True,
                'enable_json': True,
                'enable_db': False
            }
            
            signal_store = SignalStore(test_config)
            
            # Store signal
            success = signal_store.store_signal(test_signal)
            assert success, "Failed to store signal"
            
            # Retrieve signals
            retrieved = signal_store.get_signals(symbol='TEST', limit=10)
            assert len(retrieved) > 0, "No signals retrieved"
            
            # Validate retrieved signal
            retrieved_signal = retrieved[0]
            assert retrieved_signal['symbol'] == 'TEST', "Incorrect symbol in retrieved signal"
            
            logger.info("✅ Signal Store test passed")
            self.test_results['signal_store'] = True
            
            # Cleanup
            import shutil
            if Path('test_signals').exists():
                shutil.rmtree('test_signals')
                
        except Exception as e:
            logger.error(f"❌ Signal Store test failed: {e}")
            
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        try:
            logger.info("Testing End-to-End Pipeline...")
            
            # Generate comprehensive test data
            symbols = ['RELIANCE', 'TCS', 'INFY']
            timeframes = ['5m', '15m']
            
            all_signals = {}
            
            for symbol in symbols:
                for timeframe in timeframes:
                    # Generate test data
                    df = TestDataGenerator.generate_ohlcv_data(symbol, 5, timeframe)
                    df = TestDataGenerator.add_pattern_signals(df)
                    
                    # Initialize components
                    feature_engine = FeatureEngine()
                    strategy_runner = LiveStrategyRunner()
                    aggregator = SignalAggregator()
                    
                    # Process data through pipeline
                    df_with_features = feature_engine.calculate_all_features(df)
                    signals = strategy_runner.run_all_strategies(df_with_features, symbol, timeframe)
                    aggregated = aggregator.aggregate_signals(signals, symbol)
                    
                    if aggregated:
                        all_signals[f"{symbol}_{timeframe}"] = aggregated
                        
            # Initialize signal store
            test_config = {
                'base_path': 'test_pipeline_signals',
                'enable_csv': True,
                'enable_json': True,
                'enable_db': False
            }
            
            signal_store = SignalStore(test_config)
            
            # Store all signals
            total_stored = 0
            for symbol_timeframe, signals in all_signals.items():
                stored = signal_store.store_signals(signals)
                total_stored += stored
                
            logger.info(f"Stored {total_stored} signals from pipeline")
            
            # Validate pipeline output
            assert total_stored > 0, "No signals generated in end-to-end test"
            
            # Get statistics
            stats = signal_store.get_signal_statistics()
            assert stats['total_signals'] > 0, "No signals in statistics"
            
            logger.info("✅ End-to-End Pipeline test passed")
            self.test_results['end_to_end'] = True
            
            # Cleanup
            import shutil
            if Path('test_pipeline_signals').exists():
                shutil.rmtree('test_pipeline_signals')
                
        except Exception as e:
            logger.error(f"❌ End-to-End Pipeline test failed: {e}")
            
    def print_test_results(self):
        """Print test results summary"""
        logger.info("\n" + "="*50)
        logger.info("PIPELINE TEST RESULTS")
        logger.info("="*50)
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        logger.info("="*50)
        logger.info(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! Pipeline is ready for production.")
        else:
            logger.warning("⚠️  Some tests failed. Please review and fix issues.")
            
        logger.info("="*50)

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Mini-Simon Pipeline Validation...")
    
    # Run tests
    test_pipeline = PipelineTest()
    success = test_pipeline.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
