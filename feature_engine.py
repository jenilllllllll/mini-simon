"""
Feature Engine Module
Computes real-time features for SMC strategies including VWAP, wicks, volume spike detection, swing points
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class FeatureCalculator:
    """Base class for feature calculations"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features on DataFrame"""
        raise NotImplementedError

class VWAPCalculator(FeatureCalculator):
    """Volume Weighted Average Price calculator"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__(lookback_period)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related features"""
        if len(df) < 2:
            return df
            
        df = df.copy()
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP calculation
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # VWAP bands
        df['vwap_std'] = df['typical_price'].rolling(window=self.lookback_period).std()
        df['vwap_upper'] = df['vwap'] + (2 * df['vwap_std'])
        df['vwap_lower'] = df['vwap'] - (2 * df['vwap_std'])
        
        # Price distance from VWAP
        df['price_vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        return df

class VolumeFeatures(FeatureCalculator):
    """Volume-based features including spike detection"""
    
    def __init__(self, volume_lookback: int = 20, spike_threshold: float = 2.0):
        super().__init__(volume_lookback)
        self.spike_threshold = spike_threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features"""
        if len(df) < self.lookback_period:
            return df
            
        df = df.copy()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.lookback_period).mean()
        
        # Volume spike detection
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > self.spike_threshold
        
        # Volume trend
        df['volume_trend'] = df['volume'] > df['volume_ma']
        
        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=50).mean()
        
        # Price-Volume divergence
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume_divergence'] = (df['price_change'] > 0) & (df['volume_change'] < 0)
        
        return df

class WickFeatures(FeatureCalculator):
    """Wick-based features for rejection patterns"""
    
    def __init__(self, wick_threshold: float = 0.3):
        self.wick_threshold = wick_threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wick features"""
        df = df.copy()
        
        # Calculate body and wicks
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_wick'] = df['upper_wick'] + df['lower_wick']
        df['range'] = df['high'] - df['low']
        
        # Wick ratios
        df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
        df['upper_wick_ratio'] = df['upper_wick'] / (df['range'] + 1e-9)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['range'] + 1e-9)
        
        # Doji detection (small body)
        df['is_doji'] = df['body_ratio'] < 0.1
        
        # Long wick patterns
        df['long_upper_wick'] = (df['upper_wick_ratio'] > self.wick_threshold) & (df['body_ratio'] < 0.5)
        df['long_lower_wick'] = (df['lower_wick_ratio'] > self.wick_threshold) & (df['body_ratio'] < 0.5)
        
        # Wick rejection patterns
        df['bullish_rejection'] = (df['lower_wick_ratio'] > 0.4) & (df['close'] > df['open'])
        df['bearish_rejection'] = (df['upper_wick_ratio'] > 0.4) & (df['close'] < df['open'])
        
        return df

class SwingPointDetector(FeatureCalculator):
    """Swing point detection for liquidity levels"""
    
    def __init__(self, swing_lookback: int = 5, strength_threshold: int = 2):
        self.swing_lookback = swing_lookback
        self.strength_threshold = strength_threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate swing points"""
        if len(df) < self.swing_lookback * 2 + 1:
            return df
            
        df = df.copy()
        
        # Initialize swing point columns
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        # Detect swing highs and lows
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
                    
            if is_swing_high:
                df.loc[i, 'swing_high'] = True
                df.loc[i, 'swing_high_price'] = current_high
                
            # Check for swing low
            is_swing_low = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                df.loc[i, 'swing_low'] = True
                df.loc[i, 'swing_low_price'] = current_low
                
        # Calculate swing strength
        df['swing_strength'] = 0
        for i in range(len(df)):
            if df.iloc[i]['swing_high']:
                # Count how many bars to left and right are lower
                left_count = sum(1 for j in range(max(0, i - self.swing_lookback), i) 
                               if df.iloc[j]['high'] < df.iloc[i]['high'])
                right_count = sum(1 for j in range(i + 1, min(len(df), i + self.swing_lookback + 1)) 
                                if df.iloc[j]['high'] < df.iloc[i]['high'])
                df.loc[i, 'swing_strength'] = left_count + right_count
                
            elif df.iloc[i]['swing_low']:
                # Count how many bars to left and right are higher
                left_count = sum(1 for j in range(max(0, i - self.swing_lookback), i) 
                               if df.iloc[j]['low'] > df.iloc[i]['low'])
                right_count = sum(1 for j in range(i + 1, min(len(df), i + self.swing_lookback + 1)) 
                                if df.iloc[j]['low'] > df.iloc[i]['low'])
                df.loc[i, 'swing_strength'] = left_count + right_count
                
        # Filter strong swing points
        df['strong_swing_high'] = (df['swing_high']) & (df['swing_strength'] >= self.strength_threshold)
        df['strong_swing_low'] = (df['swing_low']) & (df['swing_strength'] >= self.strength_threshold)
        
        return df

class LiquiditySweepDetector(FeatureCalculator):
    """Liquidity sweep detection"""
    
    def __init__(self, swing_lookback: int = 10):
        self.swing_lookback = swing_lookback
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect liquidity sweeps"""
        if len(df) < self.swing_lookback:
            return df
            
        df = df.copy()
        
        # Get recent swing highs and lows
        df['recent_swing_high'] = np.nan
        df['recent_swing_low'] = np.nan
        
        for i in range(len(df)):
            # Look back for most recent swing points
            lookback_start = max(0, i - self.swing_lookback)
            
            swing_highs = df.loc[lookback_start:i, 'swing_high_price'].dropna()
            swing_lows = df.loc[lookback_start:i, 'swing_low_price'].dropna()
            
            if not swing_highs.empty:
                df.loc[i, 'recent_swing_high'] = swing_highs.iloc[-1]
            if not swing_lows.empty:
                df.loc[i, 'recent_swing_low'] = swing_lows.iloc[-1]
                
        # Detect liquidity sweeps
        df['liquidity_sweep_down'] = False
        df['liquidity_sweep_up'] = False
        
        for i in range(1, len(df)):
            if not pd.isna(df.iloc[i-1]['recent_swing_low']):
                # Check for downward sweep
                if df.iloc[i]['low'] < df.iloc[i-1]['recent_swing_low']:
                    df.loc[i, 'liquidity_sweep_down'] = True
                    
            if not pd.isna(df.iloc[i-1]['recent_swing_high']):
                # Check for upward sweep
                if df.iloc[i]['high'] > df.iloc[i-1]['recent_swing_high']:
                    df.loc[i, 'liquidity_sweep_up'] = True
                    
        return df

class BodyImbalanceDetector(FeatureCalculator):
    """Body imbalance detection"""
    
    def __init__(self, imbalance_threshold: float = 0.6):
        self.imbalance_threshold = imbalance_threshold
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect body imbalance patterns"""
        df = df.copy()
        
        # Calculate body position within range
        df['body_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        
        # Strong bullish body (close near high)
        df['strong_bullish_body'] = (df['close'] > df['open']) & (df['body_position'] > self.imbalance_threshold)
        
        # Strong bearish body (close near low)
        df['strong_bearish_body'] = (df['close'] < df['open']) & (df['body_position'] < (1 - self.imbalance_threshold))
        
        # Body imbalance after sweep
        df['bullish_imbalance_after_sweep'] = df['strong_bullish_body'] & df['liquidity_sweep_down'].shift(1).fillna(False)
        df['bearish_imbalance_after_sweep'] = df['strong_bearish_body'] & df['liquidity_sweep_up'].shift(1).fillna(False)
        
        return df

class ATRFeatures(FeatureCalculator):
    """ATR-based features for volatility and risk management"""
    
    def __init__(self, atr_period: int = 14):
        super().__init__(atr_period)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR features"""
        if len(df) < self.lookback_period:
            return df
            
        df = df.copy()
        
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=self.lookback_period).mean()
        
        # ATR percentage
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # ATR bands
        df['atr_upper'] = df['close'] + df['atr']
        df['atr_lower'] = df['close'] - df['atr']
        
        # Volatility regime
        df['volatility_regime'] = pd.qcut(df['atr_pct'], q=3, labels=['Low', 'Medium', 'High'])
        
        return df

class TrendFeatures(FeatureCalculator):
    """Trend analysis features"""
    
    def __init__(self, fast_ma: int = 10, slow_ma: int = 20):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features"""
        if len(df) < self.slow_ma:
            return df
            
        df = df.copy()
        
        # Moving averages
        df['ma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Trend direction
        df['trend_up'] = df['ma_fast'] > df['ma_slow']
        df['trend_down'] = df['ma_fast'] < df['ma_slow']
        
        # MA crossover signals
        df['ma_cross_up'] = df['trend_up'] & (~df['trend_up'].shift(1).fillna(False))
        df['ma_cross_down'] = df['trend_down'] & (~df['trend_down'].shift(1).fillna(False))
        
        # Price relative to MAs
        df['price_above_fast_ma'] = df['close'] > df['ma_fast']
        df['price_above_slow_ma'] = df['close'] > df['ma_slow']
        
        # MA slope
        df['ma_fast_slope'] = df['ma_fast'] - df['ma_fast'].shift(3)
        df['ma_slow_slope'] = df['ma_slow'] - df['ma_slow'].shift(3)
        
        # Trend strength
        df['trend_strength'] = abs(df['ma_fast'] - df['ma_slow']) / df['ma_slow'] * 100
        
        return df

class FeatureEngine:
    """Main feature engine that coordinates all feature calculations"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize feature calculators
        self.vwap_calc = VWAPCalculator(lookback_period=self.config.get('vwap_lookback', 20))
        self.volume_calc = VolumeFeatures(
            volume_lookback=self.config.get('volume_lookback', 20),
            spike_threshold=self.config.get('spike_threshold', 2.0)
        )
        self.wick_calc = WickFeatures(wick_threshold=self.config.get('wick_threshold', 0.3))
        self.swing_calc = SwingPointDetector(
            swing_lookback=self.config.get('swing_lookback', 5),
            strength_threshold=self.config.get('strength_threshold', 2)
        )
        self.liquidity_calc = LiquiditySweepDetector(swing_lookback=self.config.get('swing_lookback', 10))
        self.imbalance_calc = BodyImbalanceDetector(imbalance_threshold=self.config.get('imbalance_threshold', 0.6))
        self.atr_calc = ATRFeatures(atr_period=self.config.get('atr_period', 14))
        self.trend_calc = TrendFeatures(
            fast_ma=self.config.get('fast_ma', 10),
            slow_ma=self.config.get('slow_ma', 20)
        )
        
        # Feature cache for performance
        self.feature_cache = {}
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for the given DataFrame"""
        if df.empty:
            return df
            
        try:
            # Start with basic OHLCV features
            df = df.copy()
            
            # Add basic features
            df['is_bullish'] = df['close'] > df['open']
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = abs(df['price_change'])
            
            # Calculate all feature sets
            df = self.vwap_calc.calculate(df)
            df = self.volume_calc.calculate(df)
            df = self.wick_calc.calculate(df)
            df = self.swing_calc.calculate(df)
            df = self.liquidity_calc.calculate(df)
            df = self.imbalance_calc.calculate(df)
            df = self.atr_calc.calculate(df)
            df = self.trend_calc.calculate(df)
            
            # Add session information
            df = self._add_session_features(df)
            
            # Add time-based features
            df = self._add_time_features(df)
            
            logger.debug(f"Calculated {len(df.columns)} features for {len(df)} bars")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df
            
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading session features"""
        df = df.copy()
        
        # Extract hour from datetime
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        
        # Define sessions
        def get_session(hour):
            if 9 <= hour < 11:
                return 'morning'
            elif 11 <= hour < 13:
                return 'midday'
            elif 13 <= hour < 15:
                return 'afternoon'
            else:
                return 'closing'
                
        df['session'] = df['hour'].apply(get_session)
        
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        df['minute'] = pd.to_datetime(df['date']).dt.minute
        
        return df
        
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of calculated features"""
        if df.empty:
            return {}
            
        summary = {
            'total_features': len(df.columns),
            'data_points': len(df),
            'feature_categories': {
                'volume_features': [col for col in df.columns if 'volume' in col.lower()],
                'wick_features': [col for col in df.columns if 'wick' in col.lower() or 'body' in col.lower()],
                'swing_features': [col for col in df.columns if 'swing' in col.lower()],
                'liquidity_features': [col for col in df.columns if 'liquidity' in col.lower()],
                'vwap_features': [col for col in df.columns if 'vwap' in col.lower()],
                'atr_features': [col for col in df.columns if 'atr' in col.lower()],
                'trend_features': [col for col in df.columns if 'ma' in col.lower() or 'trend' in col.lower()]
            }
        }
        
        return summary
