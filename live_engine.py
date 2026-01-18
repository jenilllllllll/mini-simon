"""
Live Engine - Master Orchestrator
Coordinates all components of the real-time signal engine
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import threading
import signal as signal_module
import sys

from live_data_feed import LiveDataFeed
from rest_data_feed import RestDataFeed
from feature_engine import FeatureEngine
from live_strategy_runner import LiveStrategyRunner
from live_signal_aggregator import SignalAggregator, SignalConsolidator
from signal_store import SignalStore

logger = logging.getLogger(__name__)

class LiveEngine:
    """Master orchestrator for the real-time signal engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.data_feed = None
        self.feature_engine = None
        self.strategy_runner = None
        self.signal_aggregator = None
        self.signal_consolidator = None
        self.signal_store = None
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'signals_stored': 0,
            'errors': 0,
            'start_time': None,
            'last_signal_time': None
        }
        
        # Setup signal handlers for graceful shutdown
        signal_module.signal(signal_module.SIGINT, self._signal_handler)
        signal_module.signal(signal_module.SIGTERM, self._signal_handler)
        
    def initialize(self) -> bool:
        """Initialize all engine components"""
        try:
            logger.info("Initializing Live Engine...")
            
            # Initialize feature engine
            self.feature_engine = FeatureEngine(self.config.get('feature_engine', {}))
            logger.info("Feature engine initialized")
            
            # Initialize strategy runner
            self.strategy_runner = LiveStrategyRunner(self.config.get('strategy_runner', {}))
            logger.info("Strategy runner initialized")
            
            # Initialize signal aggregator
            self.signal_aggregator = SignalAggregator(self.config.get('signal_aggregator', {}))
            self.signal_consolidator = SignalConsolidator(self.signal_aggregator)
            logger.info("Signal aggregator initialized")
            
            # Initialize signal store
            self.signal_store = SignalStore(self.config.get('signal_store', {}))
            logger.info("Signal store initialized")
            
            # Initialize data feed
            self.data_feed = LiveDataFeed(self.config.get('data_feed', {}))
            
            # Register candle callback
            self.data_feed.add_candle_callback(self._on_new_candle)
            
            logger.info("Live Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Live Engine: {e}")
            return False
            
    def start(self) -> bool:
        """Start the live engine"""
        try:
            logger.info("Starting Live Engine...")
            
            # First try WebSocket data feed, fallback to REST API
            try:
                logger.info("Attempting WebSocket connection...")
                self.data_feed = LiveDataFeed(self.config.get('data_feed', {}))
                self.data_feed.add_callback(self._on_new_candle)
                self.data_feed.start()
                
                # Wait a bit to see if WebSocket connects
                import time
                time.sleep(5)
                
                if hasattr(self.data_feed, 'ws_feed') and self.data_feed.ws_feed.is_connected:
                    logger.info("WebSocket connection established!")
                else:
                    logger.warning("WebSocket failed to connect, switching to REST API...")
                    self.data_feed.stop()
                    raise Exception("WebSocket connection failed")
                    
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}")
                logger.info("Switching to REST API data feed...")
                
                # Use REST API fallback
                self.data_feed = RestDataFeed(self.config.get('data_feed', {}))
                self.data_feed.add_callback(self._on_new_candle)
                self.data_feed.start()
                
                # Pre-load some historical data
                self._preload_historical_data()
                
                logger.info("REST API data feed started")
            
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            logger.info("Live Engine started successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting Live Engine: {e}")
            self.stop()
            return False
            
    def stop(self):
        """Stop the live engine"""
        logger.info("Stopping Live Engine...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop data feed
        if self.data_feed:
            self.data_feed.stop()
            
        # Print final statistics
        self._print_final_stats()
        
        logger.info("Live Engine stopped")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
        sys.exit(0)
        
    def _preload_historical_data(self):
        """Pre-load historical data for all symbols and timeframes"""
        try:
            logger.info("Pre-loading historical data...")
            
            symbols = self.config.get('data_feed', {}).get('symbols', [])
            timeframes = self.config.get('data_feed', {}).get('timeframes', [])
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Get historical data
                        df = self.data_feed.get_historical_candles(symbol, timeframe, count=100)
                        if not df.empty:
                            logger.info(f"Loaded {len(df)} historical candles for {symbol} {timeframe}")
                        else:
                            logger.warning(f"No historical data available for {symbol} {timeframe}")
                            
                    except Exception as e:
                        logger.error(f"Error loading historical data for {symbol} {timeframe}: {e}")
                        
            logger.info("Historical data pre-loading completed")
            
        except Exception as e:
            logger.error(f"Error in pre-loading historical data: {e}")
            
    def _run_processing_loop(self):
        """Main processing loop"""
        logger.info("Starting main processing loop...")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Check data feed connection (REST API is always "connected")
                    if hasattr(self.data_feed, 'is_connected') and not self.data_feed.is_connected():
                        logger.warning("Data feed disconnected, attempting to reconnect...")
                        time.sleep(5)
                        continue
                        
                    # Process any pending signals
                    self._process_pending_signals()
                    
                    # Sleep to prevent high CPU usage
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.stop()
            
    def _on_new_candle(self, symbol: str, timeframe: str, candle: dict):
        """Handle new candle formation"""
        try:
            logger.debug(f"New candle: {symbol} {timeframe} at {candle['timestamp']}")
            
            # Get recent data for analysis
            df = self.data_feed.get_candles(symbol, timeframe, count=200)
            
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} bars")
                return
                
            # Calculate features
            df_with_features = self.feature_engine.calculate_all_features(df)
            
            # Run strategies
            signals = self.strategy_runner.run_all_strategies(df_with_features, symbol, timeframe)
            
            if signals:
                logger.info(f"Generated {len(signals)} signals for {symbol} {timeframe}")
                
                # Aggregate signals
                aggregated_signals = self.signal_aggregator.aggregate_signals(signals, symbol)
                
                if aggregated_signals:
                    logger.info(f"Aggregated to {len(aggregated_signals)} final signals for {symbol}")
                    
                    # Store signals
                    stored_count = self.signal_store.store_signals(aggregated_signals)
                    self.stats['signals_generated'] += len(aggregated_signals)
                    self.stats['signals_stored'] += stored_count
                    self.stats['last_signal_time'] = datetime.now()
                    
                    # Log signal details
                    for agg_signal in aggregated_signals:
                        logger.info(f"Signal: {agg_signal.symbol} {agg_signal.final_action} "
                                  f"@ {agg_signal.entry_price} (Conf: {agg_signal.aggregated_confidence:.2f})")
                        
        except Exception as e:
            logger.error(f"Error processing new candle for {symbol} {timeframe}: {e}")
            self.stats['errors'] += 1
            
    def _process_pending_signals(self):
        """Process any pending signals or perform maintenance tasks"""
        try:
            # Periodic maintenance tasks
            current_time = datetime.now()
            
            # Clean up old data every hour
            if current_time.minute == 0 and current_time.second < 5:
                days_to_keep = self.config.get('cleanup_days', 30)
                self.signal_store.cleanup_old_data(days_to_keep)
                
            # Log statistics every 10 minutes
            if current_time.minute % 10 == 0 and current_time.second < 5:
                self._log_statistics()
                
        except Exception as e:
            logger.error(f"Error in maintenance tasks: {e}")
            
    def _log_statistics(self):
        """Log current statistics"""
        try:
            uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
            
            logger.info(f"Engine Stats - Uptime: {uptime}, "
                       f"Signals Generated: {self.stats['signals_generated']}, "
                       f"Signals Stored: {self.stats['signals_stored']}, "
                       f"Errors: {self.stats['errors']}")
            
            # Get signal store statistics
            store_stats = self.signal_store.get_signal_statistics()
            logger.info(f"Store Stats - Total: {store_stats.get('total_signals', 0)}, "
                       f"Avg Confidence: {store_stats.get('average_confidence', 0):.2f}")
                       
        except Exception as e:
            logger.error(f"Error logging statistics: {e}")
            
    def _print_final_stats(self):
        """Print final statistics on shutdown"""
        try:
            if self.stats['start_time']:
                uptime = datetime.now() - self.stats['start_time']
                logger.info(f"Final Stats - Total Uptime: {uptime}")
                
            logger.info(f"Final Stats - Signals Generated: {self.stats['signals_generated']}")
            logger.info(f"Final Stats - Signals Stored: {self.stats['signals_stored']}")
            logger.info(f"Final Stats - Errors: {self.stats['errors']}")
            
        except Exception as e:
            logger.error(f"Error printing final stats: {e}")
            
    def get_engine_status(self) -> Dict:
        """Get current engine status"""
        try:
            status = {
                'is_running': self.is_running,
                'data_feed_connected': (hasattr(self.data_feed, 'is_connected') and self.data_feed.is_connected()) if self.data_feed else True,
                'uptime': str(datetime.now() - self.stats['start_time']) if self.stats['start_time'] else '0:00:00',
                'statistics': self.stats.copy(),
                'components': {
                    'data_feed': self.data_feed is not None,
                    'feature_engine': self.feature_engine is not None,
                    'strategy_runner': self.strategy_runner is not None,
                    'signal_aggregator': self.signal_aggregator is not None,
                    'signal_store': self.signal_store is not None
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return {'error': str(e)}
            
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals"""
        try:
            if not self.signal_store:
                return []
                
            # Get signals from last 24 hours
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=1)
            
            signals = self.signal_store.get_signals(
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
            
    def export_signals(self, filepath: str, format: str = 'csv', **filters) -> bool:
        """Export signals to file"""
        try:
            if not self.signal_store:
                logger.error("Signal store not initialized")
                return False
                
            return self.signal_store.export_signals(filepath, format, **filters)
            
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return False
            
    def run_health_check(self) -> Dict:
        """Run comprehensive health check"""
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {}
            }
            
            # Check data feed
            if self.data_feed:
                data_feed_status = 'connected' if self.data_feed.is_connected() else 'disconnected'
                health['checks']['data_feed'] = {
                    'status': data_feed_status,
                    'details': 'WebSocket connection status'
                }
                if data_feed_status != 'connected':
                    health['overall_status'] = 'degraded'
            else:
                health['checks']['data_feed'] = {'status': 'not_initialized', 'details': 'Data feed not initialized'}
                health['overall_status'] = 'unhealthy'
                
            # Check components
            components = {
                'feature_engine': self.feature_engine,
                'strategy_runner': self.strategy_runner,
                'signal_aggregator': self.signal_aggregator,
                'signal_store': self.signal_store
            }
            
            for name, component in components.items():
                if component:
                    health['checks'][name] = {'status': 'initialized', 'details': f'{name} initialized'}
                else:
                    health['checks'][name] = {'status': 'not_initialized', 'details': f'{name} not initialized'}
                    health['overall_status'] = 'unhealthy'
                    
            # Check signal generation
            if self.stats['last_signal_time']:
                time_since_last_signal = datetime.now() - self.stats['last_signal_time']
                if time_since_last_signal > timedelta(hours=1):
                    health['checks']['signal_generation'] = {
                        'status': 'stale',
                        'details': f'No signals for {time_since_last_signal}'
                    }
                    health['overall_status'] = 'degraded'
                else:
                    health['checks']['signal_generation'] = {
                        'status': 'active',
                        'details': f'Last signal {time_since_last_signal} ago'
                    }
            else:
                health['checks']['signal_generation'] = {
                    'status': 'no_signals',
                    'details': 'No signals generated yet'
                }
                
            # Check error rate
            if self.stats['errors'] > 10:
                health['checks']['error_rate'] = {
                    'status': 'high',
                    'details': f'{self.stats["errors"]} errors encountered'
                }
                health['overall_status'] = 'degraded'
            else:
                health['checks']['error_rate'] = {
                    'status': 'normal',
                    'details': f'{self.stats["errors"]} errors encountered'
                }
                
            return health
            
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

class EngineManager:
    """Manages engine lifecycle and provides interface for external control"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.engine = None
        self.engine_thread = None
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'data_feed': {
                'app_id': 'YOUR_APP_ID',
                'access_token': 'YOUR_ACCESS_TOKEN',
                'symbols': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],
                'timeframes': ['5m', '15m', '60m']
            },
            'feature_engine': {
                'vwap_lookback': 20,
                'volume_lookback': 20,
                'spike_threshold': 2.0,
                'swing_lookback': 5,
                'strength_threshold': 2,
                'imbalance_threshold': 0.6,
                'atr_period': 14,
                'fast_ma': 10,
                'slow_ma': 20
            },
            'strategy_runner': {
                'strategy_weights': {
                    'vol_spike': 0.35,
                    'body_imbalance': 0.25,
                    'order_block': 0.25,
                    'stock_burner': 0.15
                },
                'timeframe_weights': {
                    '3m': 0.5, '5m': 0.6, '15m': 0.7,
                    '60m': 1.0, '120m': 1.2, '180m': 1.3,
                    '240m': 1.4, '1D': 1.6
                }
            },
            'signal_aggregator': {
                'min_confidence_threshold': 0.3,
                'confluence_threshold': 2,
                'max_signals_per_symbol': 3
            },
            'signal_store': {
                'base_path': 'signals',
                'enable_csv': True,
                'enable_json': True,
                'enable_db': False,
                'daily_rotation': True
            },
            'cleanup_days': 30
        }
        
        # TODO: Load from config file if provided
        return default_config
        
    def start_engine(self) -> bool:
        """Start the engine in a separate thread"""
        try:
            if self.engine and self.engine.is_running:
                logger.warning("Engine is already running")
                return False
                
            self.engine = LiveEngine(self.config)
            
            # Start engine in separate thread
            self.engine_thread = threading.Thread(target=self.engine.start, daemon=True)
            self.engine_thread.start()
            
            # Wait for engine to start
            time.sleep(2)
            
            return self.engine.is_running
            
        except Exception as e:
            logger.error(f"Error starting engine: {e}")
            return False
            
    def stop_engine(self):
        """Stop the engine"""
        try:
            if self.engine:
                self.engine.stop()
                
            if self.engine_thread and self.engine_thread.is_alive():
                self.engine_thread.join(timeout=10)
                
            self.engine = None
            self.engine_thread = None
            
        except Exception as e:
            logger.error(f"Error stopping engine: {e}")
            
    def get_status(self) -> Dict:
        """Get engine status"""
        if not self.engine:
            return {'status': 'not_running', 'details': 'Engine not started'}
            
        return self.engine.get_engine_status()
        
    def get_health(self) -> Dict:
        """Get engine health"""
        if not self.engine:
            return {'overall_status': 'not_running'}
            
        return self.engine.run_health_check()
        
    def get_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals"""
        if not self.engine:
            return []
            
        return self.engine.get_recent_signals(limit)
        
    def export_signals(self, filepath: str, format: str = 'csv') -> bool:
        """Export signals"""
        if not self.engine:
            return False
            
        return self.engine.export_signals(filepath, format)

# Main entry point
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('live_engine.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting Mini-Simon Live Engine...")
    
    # Create and start engine manager
    manager = EngineManager()
    
    try:
        if manager.start_engine():
            logger.info("Engine started successfully")
            
            # Keep main thread alive
            while manager.engine and manager.engine.is_running:
                time.sleep(1)
                
        else:
            logger.error("Failed to start engine")
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        manager.stop_engine()
        logger.info("Engine shutdown complete")
