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

import pandas as pd

import pytz

from live_data_feed import LiveDataFeed
from rest_data_feed import RestDataFeed
from feature_engine import FeatureEngine
from live_strategy_runner import LiveStrategyRunner
from live_signal_aggregator import (
    AggregatedSignal,
    SignalAggregator,
    SignalConsolidator,
)
from signal_store import SignalStore
from mcx_symbols import get_current_commodity_symbol

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

class LiveEngine:
    """Master orchestrator for the real-time signal engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        self._last_processed_candle_ts: Dict[str, Dict[str, datetime]] = {}
        
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
            
            # Initialize data feed lazily in start() so we can choose between
            # WebSocket and REST fallback paths without creating multiple
            # instances.
            self.data_feed = None
            
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
                self.data_feed.add_candle_callback(self._on_new_candle)
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
            self.stats['start_time'] = datetime.now(IST)
            
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
                            # Actually store it in the data feed manager
                            if hasattr(self.data_feed, 'add_historical_candles'):
                                self.data_feed.add_historical_candles(symbol, timeframe, df)
                            elif hasattr(self.data_feed, 'ws_feed') and hasattr(self.data_feed.ws_feed, 'candle_manager'):
                                # For LiveDataFeed (WebSocket)
                                pass # This is already handled inside LiveDataFeed.initialize()
                                
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
            logger.info(f"Processing new {timeframe} candle for {symbol} at {candle['timestamp']}")

            candle_ts = candle.get('timestamp')
            if candle_ts is None:
                return

            # Avoid duplicate processing
            last_tf_map = self._last_processed_candle_ts.setdefault(symbol, {})
            last_ts = last_tf_map.get(timeframe)
            if last_ts is not None and last_ts == candle_ts:
                return
            last_tf_map[timeframe] = candle_ts
            
            # Get recent data for analysis
            df = self.data_feed.get_candles(symbol, timeframe, count=200)
            if df.empty or len(df) < 20:  # Reduced from 50 to 20 for faster startup
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} bars (need 20)")
                return

            # The callback fires when a new candle bucket starts. Process the most recent *closed*
            # candle by dropping the last bar if it matches the callback candle timestamp.
            try:
                if 'date' in df.columns and len(df) > 1:
                    last_bar_ts = pd.to_datetime(df['date'].iloc[-1])
                    if getattr(last_bar_ts, 'to_pydatetime', None):
                        last_bar_ts = last_bar_ts.to_pydatetime()
                    if last_bar_ts == candle_ts:
                        df = df.iloc[:-1].copy()
            except Exception:
                pass

            # Calculate features
            df_with_features = self.feature_engine.calculate_all_features(df)
            
            # Run strategies
            signals = self.strategy_runner.run_all_strategies(df_with_features, symbol, timeframe)
            
            if signals:
                logger.info(f"Raw signals generated for {symbol} {timeframe}: {len(signals)}")
            else:
                # Log periodically to show engine is alive even without signals
                if candle_ts.minute % 15 == 0:
                    logger.info(f"Engine heartbeat: Watching {symbol} {timeframe} - No strategy signals (df length: {len(df)})")

            # Only emit signals for the most recent closed candle.
            # Use a robust timestamp comparison (ignoring microseconds)
            if signals and 'date' in df_with_features.columns:
                try:
                    last_closed_ts = pd.to_datetime(df_with_features['date'].iloc[-1])
                    if getattr(last_closed_ts, 'to_pydatetime', None):
                        last_closed_ts = last_closed_ts.to_pydatetime()
                    
                    # Round last_closed_ts to nearest minute for robust matching
                    match_ts = last_closed_ts.replace(second=0, microsecond=0)
                    
                    filtered_signals = []
                    for s in signals:
                        s_ts = getattr(s, 'timestamp', None)
                        if s_ts:
                            if hasattr(s_ts, 'to_pydatetime'):
                                s_ts = s_ts.to_pydatetime()
                            # Match if same minute
                            if s_ts.replace(second=0, microsecond=0) == match_ts:
                                filtered_signals.append(s)
                    
                    signals = filtered_signals
                except Exception as e:
                    logger.error(f"Error filtering signals by timestamp: {e}")
            
            if signals:
                logger.info(f"Generated {len(signals)} signals for {symbol} {timeframe}")
                
                # Aggregate signals
                is_mcx_symbol = symbol.startswith("MCX:")
                original_min_conf = self.signal_aggregator.min_confidence_threshold
                try:
                    if is_mcx_symbol:
                        self.signal_aggregator.min_confidence_threshold = 0.0

                    aggregated_signals = self.signal_aggregator.aggregate_signals(signals, symbol)
                finally:
                    if is_mcx_symbol:
                        self.signal_aggregator.min_confidence_threshold = original_min_conf
                
                if aggregated_signals:
                    logger.info(f"Aggregated to {len(aggregated_signals)} final signals for {symbol}")
                    
                    # Store signals
                    stored_count = self.signal_store.store_signals(aggregated_signals)
                    self.stats['signals_generated'] += len(aggregated_signals)
                    self.stats['signals_stored'] += stored_count
                    self.stats['last_signal_time'] = datetime.now(IST)
                    
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
            current_time = datetime.now(IST)
            
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
            uptime = datetime.now(IST) - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
            
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
                uptime = datetime.now(IST) - self.stats['start_time']
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
                'uptime': str(datetime.now(IST) - self.stats['start_time']) if self.stats['start_time'] else '0:00:00',
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
            end_date = datetime.now(IST).date()
            start_date = end_date - timedelta(days=7)
            
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
                'timestamp': datetime.now(IST).strftime('%d-%b-%Y %I:%M %p'),
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
                time_since_last_signal = datetime.now(IST) - self.stats['last_signal_time']
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
                'timestamp': datetime.now(IST).strftime('%d-%b-%Y %I:%M %p'),
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
        """Load configuration from global config module"""
        try:
            from config import get_config
            cfg = get_config()
            
            # Map Config object to the dictionary structure expected by LiveEngine
            engine_config = {
                'data_feed': cfg.get('data_feed', {}),
                'feature_engine': cfg.get('feature_engine', {}),
                'strategy_runner': cfg.get('strategy_runner', {}),
                'signal_aggregator': cfg.get('signal_aggregator', {}),
                'signal_store': cfg.get('signal_store', {}),
                'cleanup_days': cfg.get('cleanup_days', 30)
            }
            
            # Ensure critical thresholds are copied if not in sub-configs
            if 'min_confidence_threshold' not in engine_config['signal_aggregator']:
                engine_config['signal_aggregator']['min_confidence_threshold'] = cfg.get('signal_aggregator.min_confidence_threshold', 0.2)
            
            if 'confluence_threshold' not in engine_config['signal_aggregator']:
                engine_config['signal_aggregator']['confluence_threshold'] = cfg.get('signal_aggregator.confluence_threshold', 1)

            logger.info("Engine configuration loaded from config.py")
            return engine_config
            
        except Exception as e:
            logger.error(f"Error loading config from config.py: {e}. Using fallback defaults.")
            # Fallback defaults if config.py fails
            return {
                'data_feed': {'symbols': [], 'timeframes': ['5m', '15m']},
                'signal_aggregator': {'min_confidence_threshold': 0.2, 'confluence_threshold': 1},
                'signal_store': {'base_path': 'signals', 'enable_json': True},
                'cleanup_days': 30
            }
        
    def start_engine(self) -> bool:
        """Start the engine in a separate thread"""
        try:
            if self.engine and self.engine.is_running:
                logger.warning("Engine is already running")
                return False
                
            self.engine = LiveEngine(self.config)

            if not self.engine.initialize():
                logger.error("Engine initialization failed; not starting engine thread")
                self.engine = None
                return False

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
