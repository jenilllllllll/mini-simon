"""
Utility Functions Module
Common utility functions used across the live trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Union, Any
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TimeUtils:
    """Time-related utility functions"""
    
    @staticmethod
    def is_market_hours(dt: datetime = None) -> bool:
        """Check if given time is within Indian market hours"""
        if dt is None:
            dt = datetime.now()
            
        # Check if it's a weekday (Monday=0 to Friday=4)
        if dt.weekday() > 4:
            return False
            
        # Check market hours (9:15 to 15:30)
        market_open = time(9, 15)
        market_close = time(15, 30)
        current_time = dt.time()
        
        return market_open <= current_time <= market_close
        
    @staticmethod
    def get_next_market_open(dt: datetime = None) -> datetime:
        """Get next market open time"""
        if dt is None:
            dt = datetime.now()
            
        # If it's weekend, go to Monday
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            days_until_monday = (7 - dt.weekday()) % 7 or 7
            next_open = dt + timedelta(days=days_until_monday)
            next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # If it's after market close, go to next day
            if dt.time() > time(15, 30):
                next_open = dt + timedelta(days=1)
                next_open = next_open.replace(hour=9, minute=15, second=0, microsecond=0)
            else:
                # If it's before market open, use today's open
                if dt.time() < time(9, 15):
                    next_open = dt.replace(hour=9, minute=15, second=0, microsecond=0)
                else:
                    # Market is open now
                    next_open = dt
                    
        return next_open
        
    @staticmethod
    def get_timeframe_seconds(timeframe: str) -> int:
        """Convert timeframe string to seconds"""
        timeframe_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900,
            '60m': 3600, '120m': 7200, '180m': 10800, '240m': 14400, '1D': 86400
        }
        return timeframe_map.get(timeframe, 300)  # Default to 5 minutes
        
    @staticmethod
    def align_to_timeframe(dt: datetime, timeframe: str) -> datetime:
        """Align datetime to timeframe boundary"""
        tf_seconds = TimeUtils.get_timeframe_seconds(timeframe)
        
        if timeframe == '1D':
            # For daily, align to 9:15 AM
            return dt.replace(hour=9, minute=15, second=0, microsecond=0)
        else:
            # For intraday, round down to timeframe boundary
            seconds_since_midnight = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
            aligned_seconds = int(seconds_since_midnight / tf_seconds) * tf_seconds
            aligned_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=aligned_seconds)
            return aligned_dt

class DataFrameUtils:
    """DataFrame utility functions"""
    
    @staticmethod
    def ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has required OHLCV columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'date']
        
        df = df.copy()
        
        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Convert date column
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        return df
        
    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features"""
        df = df.copy()
        
        # Basic price features
        df['is_bullish'] = df['close'] > df['open']
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = abs(df['price_change'])
        
        # Body and wicks
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df[['open', 'close']].max(axis=1) - df['high']
        df['lower_wick'] = df['low'] - df[['open', 'close']].min(axis=1)
        df['range'] = df['high'] - df['low']
        
        # Avoid division by zero
        df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
        df['upper_wick_ratio'] = df['upper_wick'] / (df['range'] + 1e-9)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['range'] + 1e-9)
        
        return df
        
    @staticmethod
    def calculate_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate various return metrics"""
        df = df.copy()
        
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
        
    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe"""
        try:
            df = df.copy()
            df.set_index('date', inplace=True)
            
            # Define resampling rule
            if timeframe == '1D':
                rule = '1D'
            else:
                rule = f'{timeframe[:-1]}{timeframe[-1].upper()}'  # e.g., '5T' for 5m
                
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            resampled.reset_index(inplace=True)
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df

class MathUtils:
    """Mathematical utility functions"""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for division by zero"""
        try:
            if abs(denominator) < 1e-9:
                return default
            return numerator / denominator
        except Exception:
            return default
            
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
        
    @staticmethod
    def normalize_series(series: pd.Series, method: str = 'minmax') -> pd.Series:
        """Normalize a pandas Series"""
        if method == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif method == 'zscore':
            return (series - series.mean()) / series.std()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

class FileUtils:
    """File utility functions"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    @staticmethod
    def safe_json_load(filepath: Union[str, Path], default: Any = None) -> Any:
        """Safely load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {e}")
            return default
            
    @staticmethod
    def safe_json_save(data: Any, filepath: Union[str, Path]) -> bool:
        """Safely save data to JSON file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}")
            return False
            
    @staticmethod
    def get_file_size(filepath: Union[str, Path]) -> int:
        """Get file size in bytes"""
        try:
            return Path(filepath).stat().st_size
        except Exception:
            return 0
            
    @staticmethod
    def cleanup_old_files(directory: Union[str, Path], pattern: str = "*", days_old: int = 30) -> int:
        """Clean up old files in directory"""
        try:
            directory = Path(directory)
            if not directory.exists():
                return 0
                
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for filepath in directory.glob(pattern):
                if filepath.is_file() and datetime.fromtimestamp(filepath.stat().st_mtime) < cutoff_date:
                    filepath.unlink()
                    deleted_count += 1
                    
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
            return 0

class ValidationUtils:
    """Validation utility functions"""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> List[str]:
        """Validate OHLCV data and return list of issues"""
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
            
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            
        # Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
                
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = df['high'] < df[['open', 'low', 'close']].max(axis=1)
            invalid_low = df['low'] > df[['open', 'high', 'close']].min(axis=1)
            
            if invalid_high.any():
                issues.append(f"Invalid high values in {invalid_high.sum()} rows")
            if invalid_low.any():
                issues.append(f"Invalid low values in {invalid_low.sum()} rows")
                
        # Check for NaN values
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            issues.append(f"NaN values found in: {nan_cols}")
            
        # Check for duplicates
        if 'date' in df.columns and df['date'].duplicated().any():
            issues.append(f"Duplicate dates found: {df['date'].duplicated().sum()}")
            
        return issues
        
    @staticmethod
    def validate_price_data(price: float) -> bool:
        """Validate single price value"""
        try:
            return isinstance(price, (int, float)) and price > 0 and not np.isnan(price) and not np.isinf(price)
        except Exception:
            return False
            
    @staticmethod
    def validate_signal_data(signal: Dict) -> List[str]:
        """Validate signal data dictionary"""
        issues = []
        required_fields = [
            'symbol', 'action', 'entry_price', 'stop_loss_level', 
            'target_level', 'aggregated_confidence', 'timestamp_generated'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                issues.append(f"Missing required field: {field}")
                
        # Validate action
        if 'action' in signal and signal['action'] not in ['BUY', 'SELL', 'NEUTRAL']:
            issues.append(f"Invalid action: {signal['action']}")
            
        # Validate prices
        price_fields = ['entry_price', 'stop_loss_level', 'target_level']
        for field in price_fields:
            if field in signal and not ValidationUtils.validate_price_data(signal[field]):
                issues.append(f"Invalid price for {field}: {signal[field]}")
                
        # Validate confidence
        if 'aggregated_confidence' in signal:
            conf = signal['aggregated_confidence']
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                issues.append(f"Invalid confidence: {conf}")
                
        return issues

class PerformanceUtils:
    """Performance monitoring utilities"""
    
    @staticmethod
    def calculate_performance_metrics(trades_df: pd.DataFrame) -> Dict:
        """Calculate standard performance metrics"""
        if trades_df.empty:
            return {}
            
        metrics = {}
        
        # Basic metrics
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['points'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['points'] < 0])
        
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
            
            # Profit metrics
            total_profit = trades_df['points'].sum()
            metrics['total_profit'] = total_profit
            metrics['average_profit_per_trade'] = total_profit / metrics['total_trades']
            
            # Risk metrics
            metrics['max_drawdown'] = PerformanceUtils._calculate_max_drawdown(trades_df['points'])
            metrics['profit_factor'] = PerformanceUtils._calculate_profit_factor(trades_df)
            
        return metrics
        
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = returns.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        return drawdown.min()
        
    @staticmethod
    def _calculate_profit_factor(trades_df: pd.DataFrame) -> float:
        """Calculate profit factor"""
        wins = trades_df[trades_df['points'] > 0]['points'].sum()
        losses = abs(trades_df[trades_df['points'] < 0]['points'].sum())
        
        return wins / losses if losses > 0 else float('inf')

class ConfigUtils:
    """Configuration utility functions"""
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """Deep merge two configuration dictionaries"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigUtils.merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    @staticmethod
    def get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
        """Get nested value using dot notation"""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    @staticmethod
    def set_nested_value(config: Dict, key_path: str, value: Any):
        """Set nested value using dot notation"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value

# Export all utility classes
__all__ = [
    'TimeUtils', 'DataFrameUtils', 'MathUtils', 'FileUtils',
    'ValidationUtils', 'PerformanceUtils', 'ConfigUtils'
]
