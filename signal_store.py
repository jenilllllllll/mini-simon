"""
Signal Store Module
Stores every generated signal with complete historical storage support
Supports CSV, JSON, and optional database storage with daily rotation
"""

import os
import json
import csv
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Union
import threading
import time
from contextlib import contextmanager

from live_signal_aggregator import AggregatedSignal

logger = logging.getLogger(__name__)

class SignalStore:
    """Production-grade signal storage with multiple format support"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Storage paths
        self.base_path = Path(self.config.get('base_path', 'signals'))
        self.base_path.mkdir(exist_ok=True, parents=True)
        
        # Storage settings
        self.enable_csv = self.config.get('enable_csv', True)
        self.enable_json = self.config.get('enable_json', True)
        self.enable_db = self.config.get('enable_db', False)
        
        # Daily rotation settings
        self.daily_rotation = self.config.get('daily_rotation', True)
        self.current_date = date.today()
        
        # File naming patterns
        self.csv_pattern = self.config.get('csv_pattern', 'signals_{date}.csv')
        self.json_pattern = self.config.get('json_pattern', 'signals_{date}.json')
        self.db_path = self.base_path / self.config.get('db_filename', 'signals.db')
        
        # Thread safety
        self.write_lock = threading.Lock()
        
        # Initialize database if enabled
        if self.enable_db:
            self._init_database()
            
        # Create daily directories
        if self.daily_rotation:
            self._create_daily_directories()
            
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss_level REAL NOT NULL,
                        target_level REAL NOT NULL,
                        signal_timestamp TEXT NOT NULL,
                        signal_timeframe TEXT NOT NULL,
                        chart_anchor_price REAL,
                        timestamp_generated TEXT NOT NULL,
                        aggregated_confidence REAL NOT NULL,
                        final_action TEXT NOT NULL,
                        contributing_strategies TEXT,  -- JSON array
                        contributing_timeframes TEXT,  -- JSON array
                        decision_score REAL NOT NULL,
                        version_id TEXT NOT NULL,
                        raw_strategy_outputs TEXT,  -- JSON
                        metadata TEXT,  -- JSON
                        date_created TEXT DEFAULT CURRENT_TIMESTAMP,
                        INDEX(symbol),
                        INDEX(final_action),
                        INDEX(timestamp_generated),
                        INDEX(decision_score)
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_final_action ON signals(final_action)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp_generated)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_score ON signals(decision_score)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def _create_daily_directories(self):
        """Create daily directories for file storage"""
        today_str = self.current_date.strftime('%Y-%m-%d')
        daily_path = self.base_path / today_str
        daily_path.mkdir(exist_ok=True, parents=True)
        
    def _get_daily_filepath(self, pattern: str) -> Path:
        """Get daily file path based on pattern"""
        date_str = self.current_date.strftime('%Y-%m-%d')
        filename = pattern.format(date=date_str)
        
        if self.daily_rotation:
            daily_path = self.base_path / date_str
            daily_path.mkdir(exist_ok=True, parents=True)
            return daily_path / filename
        else:
            return self.base_path / filename
            
    def _get_db_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')  # Better concurrent access
        conn.execute('PRAGMA synchronous=NORMAL')  # Balance between safety and performance
        return conn
        
    @contextmanager
    def _atomic_write(self, filepath: Path, mode: str = 'w'):
        """Context manager for atomic file writing"""
        temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
        
        try:
            with open(temp_path, mode) as f:
                yield f
                
            # Atomic move
            temp_path.replace(filepath)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e
            
    def store_signal(self, signal: AggregatedSignal) -> bool:
        """Store a single signal in all enabled formats"""
        success = True
        
        with self.write_lock:
            try:
                # Check if we need to rotate to new day
                if self.daily_rotation and date.today() != self.current_date:
                    self._rotate_to_new_day()
                    
                # Store in CSV
                if self.enable_csv:
                    success &= self._store_csv(signal)
                    
                # Store in JSON
                if self.enable_json:
                    success &= self._store_json(signal)
                    
                # Store in database
                if self.enable_db:
                    success &= self._store_db(signal)
                    
                if success:
                    logger.debug(f"Stored signal for {signal.symbol} {signal.action}")
                    
            except Exception as e:
                logger.error(f"Error storing signal: {e}")
                success = False
                
        return success
        
    def store_signals(self, signals: List[AggregatedSignal]) -> int:
        """Store multiple signals efficiently"""
        if not signals:
            return 0
            
        successful_stores = 0
        
        with self.write_lock:
            try:
                # Check for day rotation
                if self.daily_rotation and date.today() != self.current_date:
                    self._rotate_to_new_day()
                    
                # Batch store in CSV
                if self.enable_csv:
                    successful_stores += self._store_csv_batch(signals)
                    
                # Batch store in JSON
                if self.enable_json:
                    successful_stores += self._store_json_batch(signals)
                    
                # Batch store in database
                if self.enable_db:
                    successful_stores += self._store_db_batch(signals)
                    
                logger.info(f"Stored {successful_stores} signals")
                
            except Exception as e:
                logger.error(f"Error batch storing signals: {e}")
                
        return successful_stores
        
    def _store_csv(self, signal: AggregatedSignal) -> bool:
        """Store signal in CSV file"""
        try:
            filepath = self._get_daily_filepath(self.csv_pattern)
            signal_dict = signal.to_dict()
            
            # Flatten nested fields for CSV
            signal_dict['contributing_strategies'] = ','.join(signal_dict['contributing_strategies'])
            signal_dict['contributing_timeframes'] = ','.join(signal_dict['contributing_timeframes'])
            signal_dict['raw_strategy_outputs'] = json.dumps(signal_dict['raw_strategy_outputs'])
            signal_dict['metadata'] = json.dumps(signal_dict['metadata'])
            
            # Define CSV columns
            csv_columns = [
                'symbol', 'action', 'confidence', 'entry_price', 'stop_loss_level',
                'target_level', 'signal_timestamp', 'signal_timeframe', 'chart_anchor_price',
                'timestamp_generated', 'aggregated_confidence', 'final_action',
                'contributing_strategies', 'contributing_timeframes', 'decision_score',
                'version_id', 'raw_strategy_outputs', 'metadata'
            ]
            
            # Check if file exists to determine if we need headers
            file_exists = filepath.exists()
            
            with self._atomic_write(filepath, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                
                if not file_exists:
                    writer.writeheader()
                    
                # Write only the columns we need
                row_data = {col: signal_dict.get(col, '') for col in csv_columns}
                writer.writerow(row_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing signal in CSV: {e}")
            return False
            
    def _store_csv_batch(self, signals: List[AggregatedSignal]) -> int:
        """Store multiple signals in CSV file"""
        try:
            if not signals:
                return 0
                
            filepath = self._get_daily_filepath(self.csv_pattern)
            
            # Prepare data
            csv_columns = [
                'symbol', 'action', 'confidence', 'entry_price', 'stop_loss_level',
                'target_level', 'signal_timestamp', 'signal_timeframe', 'chart_anchor_price',
                'timestamp_generated', 'aggregated_confidence', 'final_action',
                'contributing_strategies', 'contributing_timeframes', 'decision_score',
                'version_id', 'raw_strategy_outputs', 'metadata'
            ]
            
            rows = []
            for signal in signals:
                signal_dict = signal.to_dict()
                signal_dict['contributing_strategies'] = ','.join(signal_dict['contributing_strategies'])
                signal_dict['contributing_timeframes'] = ','.join(signal_dict['contributing_timeframes'])
                signal_dict['raw_strategy_outputs'] = json.dumps(signal_dict['raw_strategy_outputs'])
                signal_dict['metadata'] = json.dumps(signal_dict['metadata'])
                
                row_data = {col: signal_dict.get(col, '') for col in csv_columns}
                rows.append(row_data)
                
            file_exists = filepath.exists()
            
            with self._atomic_write(filepath, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                
                if not file_exists:
                    writer.writeheader()
                    
                writer.writerows(rows)
                
            return len(signals)
            
        except Exception as e:
            logger.error(f"Error batch storing signals in CSV: {e}")
            return 0
            
    def _store_json(self, signal: AggregatedSignal) -> bool:
        """Store signal in JSON file (append mode)"""
        try:
            filepath = self._get_daily_filepath(self.json_pattern)
            
            # Read existing data if file exists
            if filepath.exists():
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
                
            # Append new signal
            data.append(signal.to_dict())
            
            # Write back atomically
            with self._atomic_write(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing signal in JSON: {e}")
            return False
            
    def _store_json_batch(self, signals: List[AggregatedSignal]) -> int:
        """Store multiple signals in JSON file"""
        try:
            if not signals:
                return 0
                
            filepath = self._get_daily_filepath(self.json_pattern)
            
            # Read existing data
            if filepath.exists():
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
                
            # Append new signals
            data.extend([signal.to_dict() for signal in signals])
            
            # Write back atomically
            with self._atomic_write(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            return len(signals)
            
        except Exception as e:
            logger.error(f"Error batch storing signals in JSON: {e}")
            return 0
            
    def _store_db(self, signal: AggregatedSignal) -> bool:
        """Store signal in SQLite database"""
        try:
            with self._get_db_connection() as conn:
                signal_dict = signal.to_dict()
                
                conn.execute('''
                    INSERT INTO signals (
                        symbol, action, confidence, entry_price, stop_loss_level,
                        target_level, signal_timestamp, signal_timeframe, chart_anchor_price,
                        timestamp_generated, aggregated_confidence, final_action,
                        contributing_strategies, contributing_timeframes, decision_score,
                        version_id, raw_strategy_outputs, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_dict['symbol'],
                    signal_dict['action'],
                    signal_dict['confidence'],
                    signal_dict['entry_price'],
                    signal_dict['stop_loss_level'],
                    signal_dict['target_level'],
                    signal_dict['signal_timestamp'],
                    signal_dict['signal_timeframe'],
                    signal_dict['chart_anchor_price'],
                    signal_dict['timestamp_generated'],
                    signal_dict['aggregated_confidence'],
                    signal_dict['final_action'],
                    json.dumps(signal_dict['contributing_strategies']),
                    json.dumps(signal_dict['contributing_timeframes']),
                    signal_dict['decision_score'],
                    signal_dict['version_id'],
                    json.dumps(signal_dict['raw_strategy_outputs']),
                    json.dumps(signal_dict['metadata'])
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error storing signal in database: {e}")
            return False
            
    def _store_db_batch(self, signals: List[AggregatedSignal]) -> int:
        """Store multiple signals in database"""
        try:
            if not signals:
                return 0
                
            with self._get_db_connection() as conn:
                data = []
                for signal in signals:
                    signal_dict = signal.to_dict()
                    data.append((
                        signal_dict['symbol'],
                        signal_dict['action'],
                        signal_dict['confidence'],
                        signal_dict['entry_price'],
                        signal_dict['stop_loss_level'],
                        signal_dict['target_level'],
                        signal_dict['signal_timestamp'],
                        signal_dict['signal_timeframe'],
                        signal_dict['chart_anchor_price'],
                        signal_dict['timestamp_generated'],
                        signal_dict['aggregated_confidence'],
                        signal_dict['final_action'],
                        json.dumps(signal_dict['contributing_strategies']),
                        json.dumps(signal_dict['contributing_timeframes']),
                        signal_dict['decision_score'],
                        signal_dict['version_id'],
                        json.dumps(signal_dict['raw_strategy_outputs']),
                        json.dumps(signal_dict['metadata'])
                    ))
                    
                conn.executemany('''
                    INSERT INTO signals (
                        symbol, action, confidence, entry_price, stop_loss_level,
                        target_level, signal_timestamp, signal_timeframe, chart_anchor_price,
                        timestamp_generated, aggregated_confidence, final_action,
                        contributing_strategies, contributing_timeframes, decision_score,
                        version_id, raw_strategy_outputs, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data)
                
                conn.commit()
                return len(signals)
                
        except Exception as e:
            logger.error(f"Error batch storing signals in database: {e}")
            return 0
            
    def _rotate_to_new_day(self):
        """Rotate storage to new day"""
        try:
            self.current_date = date.today()
            self._create_daily_directories()
            logger.info(f"Rotated to new day: {self.current_date}")
            
        except Exception as e:
            logger.error(f"Error rotating to new day: {e}")
            
    def get_signals(self, 
                   symbol: str = None,
                   start_date: date = None,
                   end_date: date = None,
                   action: str = None,
                   min_confidence: float = None,
                   limit: int = 1000) -> List[Dict]:
        """Retrieve signals with filtering options"""
        try:
            if self.enable_db:
                return self._get_signals_db(symbol, start_date, end_date, action, min_confidence, limit)
            else:
                return self._get_signals_files(symbol, start_date, end_date, action, min_confidence, limit)
                
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            return []
            
    def _get_signals_db(self, 
                       symbol: str = None,
                       start_date: date = None,
                       end_date: date = None,
                       action: str = None,
                       min_confidence: float = None,
                       limit: int = 1000) -> List[Dict]:
        """Retrieve signals from database"""
        try:
            with self._get_db_connection() as conn:
                query = "SELECT * FROM signals WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                    
                if start_date:
                    query += " AND date(timestamp_generated) >= ?"
                    params.append(start_date.strftime('%Y-%m-%d'))
                    
                if end_date:
                    query += " AND date(timestamp_generated) <= ?"
                    params.append(end_date.strftime('%Y-%m-%d'))
                    
                if action:
                    query += " AND final_action = ?"
                    params.append(action)
                    
                if min_confidence:
                    query += " AND aggregated_confidence >= ?"
                    params.append(min_confidence)
                    
                query += " ORDER BY timestamp_generated DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                signals = []
                for row in rows:
                    signal_dict = dict(row)
                    # Parse JSON fields
                    signal_dict['contributing_strategies'] = json.loads(signal_dict['contributing_strategies'])
                    signal_dict['contributing_timeframes'] = json.loads(signal_dict['contributing_timeframes'])
                    signal_dict['raw_strategy_outputs'] = json.loads(signal_dict['raw_strategy_outputs'])
                    signal_dict['metadata'] = json.loads(signal_dict['metadata'])
                    signals.append(signal_dict)
                    
                return signals
                
        except Exception as e:
            logger.error(f"Error retrieving signals from database: {e}")
            return []
            
    def _get_signals_files(self, 
                          symbol: str = None,
                          start_date: date = None,
                          end_date: date = None,
                          action: str = None,
                          min_confidence: float = None,
                          limit: int = 1000) -> List[Dict]:
        """Retrieve signals from JSON files"""
        try:
            signals = []
            
            # Determine date range to scan
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    signals.extend(self._read_date_signals(current_date))
                    current_date += timedelta(days=1)
            else:
                # Read recent dates
                for days_back in range(min(30, limit // 10)):  # Scan last 30 days or until we have enough
                    current_date = date.today() - timedelta(days=days_back)
                    signals.extend(self._read_date_signals(current_date))
                    
            # Apply filters
            filtered_signals = signals
            
            if symbol:
                filtered_signals = [s for s in filtered_signals if s['symbol'] == symbol]
                
            if action:
                filtered_signals = [s for s in filtered_signals if s['final_action'] == action]
                
            if min_confidence:
                filtered_signals = [s for s in filtered_signals if s['aggregated_confidence'] >= min_confidence]
                
            # Sort by timestamp and limit
            filtered_signals.sort(key=lambda x: x['timestamp_generated'], reverse=True)
            
            return filtered_signals[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving signals from files: {e}")
            return []
            
    def _read_date_signals(self, date_to_read: date) -> List[Dict]:
        """Read signals for a specific date"""
        try:
            date_str = date_to_read.strftime('%Y-%m-%d')
            filepath = self.base_path / date_str / self.json_pattern.format(date=date_str)
            
            if not filepath.exists():
                return []
                
            with open(filepath, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error reading signals for {date_to_read}: {e}")
            return []
            
    def get_signal_statistics(self) -> Dict:
        """Get statistics about stored signals"""
        try:
            if self.enable_db:
                return self._get_statistics_db()
            else:
                return self._get_statistics_files()
                
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}
            
    def _get_statistics_db(self) -> Dict:
        """Get statistics from database"""
        try:
            with self._get_db_connection() as conn:
                stats = {}
                
                # Total signals
                cursor = conn.execute("SELECT COUNT(*) as count FROM signals")
                stats['total_signals'] = cursor.fetchone()['count']
                
                # Signals by action
                cursor = conn.execute("""
                    SELECT final_action, COUNT(*) as count 
                    FROM signals 
                    GROUP BY final_action
                """)
                stats['by_action'] = dict(cursor.fetchall())
                
                # Signals by symbol (top 10)
                cursor = conn.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM signals 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                stats['top_symbols'] = dict(cursor.fetchall())
                
                # Average confidence
                cursor = conn.execute("SELECT AVG(aggregated_confidence) as avg_conf FROM signals")
                stats['average_confidence'] = cursor.fetchone()['avg_conf']
                
                # Date range
                cursor = conn.execute("""
                    SELECT MIN(timestamp_generated) as min_date, 
                           MAX(timestamp_generated) as max_date 
                    FROM signals
                """)
                row = cursor.fetchone()
                stats['date_range'] = {
                    'start': row['min_date'],
                    'end': row['max_date']
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
            
    def _get_statistics_files(self) -> Dict:
        """Get statistics from files"""
        try:
            stats = {
                'total_signals': 0,
                'by_action': {},
                'top_symbols': {},
                'average_confidence': 0,
                'date_range': {'start': None, 'end': None}
            }
            
            # Scan last 30 days
            all_signals = []
            for days_back in range(30):
                current_date = date.today() - timedelta(days=days_back)
                date_signals = self._read_date_signals(current_date)
                all_signals.extend(date_signals)
                
            if not all_signals:
                return stats
                
            # Calculate statistics
            stats['total_signals'] = len(all_signals)
            
            # By action
            for signal in all_signals:
                action = signal['final_action']
                stats['by_action'][action] = stats['by_action'].get(action, 0) + 1
                
            # Top symbols
            symbol_counts = {}
            for signal in all_signals:
                symbol = signal['symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
            stats['top_symbols'] = dict(sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Average confidence
            confidences = [s['aggregated_confidence'] for s in all_signals]
            stats['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            
            # Date range
            timestamps = [s['timestamp_generated'] for s in all_signals]
            if timestamps:
                stats['date_range']['start'] = min(timestamps)
                stats['date_range']['end'] = max(timestamps)
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting file statistics: {e}")
            return {}
            
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up data older than specified days"""
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            
            # Clean up database
            if self.enable_db:
                with self._get_db_connection() as conn:
                    cursor = conn.execute("DELETE FROM signals WHERE date(timestamp_generated) < ?", (cutoff_date.strftime('%Y-%m-%d'),))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    logger.info(f"Deleted {deleted_count} old signals from database")
                    
            # Clean up files
            if self.daily_rotation:
                for item in self.base_path.iterdir():
                    if item.is_dir():
                        try:
                            folder_date = datetime.strptime(item.name, '%Y-%m-%d').date()
                            if folder_date < cutoff_date:
                                import shutil
                                shutil.rmtree(item)
                                logger.info(f"Deleted old signal directory: {item}")
                        except ValueError:
                            # Not a date folder, skip
                            pass
                            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def export_signals(self, filepath: str, format: str = 'csv', **filters) -> bool:
        """Export signals to specified format"""
        try:
            signals = self.get_signals(**filters)
            
            if format.lower() == 'csv':
                df = pd.DataFrame(signals)
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(signals, f, indent=2, default=str)
            elif format.lower() == 'excel':
                df = pd.DataFrame(signals)
                df.to_excel(filepath, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
            logger.info(f"Exported {len(signals)} signals to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return False
