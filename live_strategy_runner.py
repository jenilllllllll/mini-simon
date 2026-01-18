"""
Live Strategy Runner Module
Wraps existing strategies for real-time execution without modifying their core logic
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import importlib
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class StrategySignal:
    """Standardized signal format for all strategies"""
    
    def __init__(self, 
                 strategy_name: str,
                 symbol: str,
                 timeframe: str,
                 action: str,
                 entry_price: float,
                 stop_loss: float,
                 target: float,
                 confidence: float,
                 anchor_price: float = None,
                 metadata: Dict = None):
        
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe
        self.action = action.upper()  # BUY, SELL, NEUTRAL
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.confidence = confidence
        self.anchor_price = anchor_price
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'action': self.action,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'confidence': self.confidence,
            'anchor_price': self.anchor_price,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

class VolumeSpikeLiquiditySweepWrapper:
    """Wrapper for Volume Spike + Liquidity Sweep strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "vol_spike"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run strategy on provided data"""
        signals = []
        
        try:
            if len(df) < 50:  # Need minimum data
                return signals
                
            df = df.copy()
            
            # Volume spike detection
            volume_lookback = self.config.get('volume_lookback', 20)
            spike_threshold = self.config.get('spike_threshold', 2.0)
            
            df['volume_ma'] = df['volume'].rolling(window=volume_lookback).mean()
            df['volume_spike'] = df['volume'] > (df['volume_ma'] * spike_threshold)
            
            # Swing point detection (simplified)
            swing_lookback = 5
            df['swing_high'] = False
            df['swing_low'] = False
            
            for i in range(swing_lookback, len(df) - swing_lookback):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check swing high
                is_swing_high = all(df.iloc[j]['high'] < current_high 
                                  for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
                if is_swing_high:
                    df.loc[i, 'swing_high'] = True
                    
                # Check swing low
                is_swing_low = all(df.iloc[j]['low'] > current_low 
                                  for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
                if is_swing_low:
                    df.loc[i, 'swing_low'] = True
                    
            # Liquidity sweep detection
            df['liquidity_sweep_down'] = False
            df['liquidity_sweep_up'] = False
            
            for i in range(1, len(df)):
                if df.iloc[i-1]['swing_low'] and df.iloc[i]['low'] < df.iloc[i-1]['low']:
                    df.loc[i, 'liquidity_sweep_down'] = True
                    
                if df.iloc[i-1]['swing_high'] and df.iloc[i]['high'] > df.iloc[i-1]['high']:
                    df.loc[i, 'liquidity_sweep_up'] = True
                    
            # Generate signals
            for i in range(2, len(df)):
                if (df.iloc[i]['volume_spike'] and 
                    df.iloc[i-1]['liquidity_sweep_down'] and 
                    df.iloc[i]['is_bullish']):
                    
                    # Buy signal
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['low']
                    target = entry_price + 2 * (entry_price - stop_loss)
                    confidence = min(0.8, df.iloc[i]['volume_ratio'] / 3.0)
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i-1]['swing_low_price'] if 'swing_low_price' in df.columns else df.iloc[i-1]['low']
                    )
                    signals.append(signal)
                    
                elif (df.iloc[i]['volume_spike'] and 
                      df.iloc[i-1]['liquidity_sweep_up'] and 
                      not df.iloc[i]['is_bullish']):
                    
                    # Sell signal
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['high']
                    target = entry_price - 2 * (stop_loss - entry_price)
                    confidence = min(0.8, df.iloc[i]['volume_ratio'] / 3.0)
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i-1]['swing_high_price'] if 'swing_high_price' in df.columns else df.iloc[i-1]['high']
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error in Volume Spike strategy: {e}")
            
        return signals

class BodyImbalanceWrapper:
    """Wrapper for Body Imbalance after Liquidity Sweep strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "body_imbalance"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run strategy on provided data"""
        signals = []
        
        try:
            if len(df) < 50:
                return signals
                
            df = df.copy()
            
            # Body imbalance detection
            df['body_ratio'] = df['body'] / (df['upper_wick'] + df['lower_wick'] + 1e-9)
            
            # Liquidity sweep detection (simplified)
            swing_lookback = 3
            for i in range(swing_lookback, len(df) - swing_lookback):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']
                
                # Check swing high
                is_swing_high = all(df.iloc[j]['high'] < current_high 
                                  for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
                
                # Check swing low
                is_swing_low = all(df.iloc[j]['low'] > current_low 
                                  for j in range(i - swing_lookback, i + swing_lookback + 1) if j != i)
                
                if is_swing_high:
                    df.loc[i, 'swing_high'] = True
                    df.loc[i, 'swing_high_price'] = current_high
                    
                if is_swing_low:
                    df.loc[i, 'swing_low'] = True
                    df.loc[i, 'swing_low_price'] = current_low
                    
            # Liquidity sweep detection
            df['liquidity_sweep_down'] = False
            df['liquidity_sweep_up'] = False
            
            for i in range(1, len(df)):
                if df.iloc[i-1]['swing_low'] and df.iloc[i]['low'] < df.iloc[i-1]['low']:
                    df.loc[i, 'liquidity_sweep_down'] = True
                    
                if df.iloc[i-1]['swing_high'] and df.iloc[i]['high'] > df.iloc[i-1]['high']:
                    df.loc[i, 'liquidity_sweep_up'] = True
                    
            # Generate signals based on body imbalance after sweep
            for i in range(2, len(df)):
                curr = df.iloc[i]
                prev = df.iloc[i-1]
                
                # Common filters
                vol_confirm = curr.get('volume_spike', True)
                is_valid_session = curr.get('session', 'morning') in ['morning', 'afternoon']
                is_not_inside_bar = not ((curr['high'] < prev['high']) and (curr['low'] > prev['low']))
                
                # Long setup
                is_liquidity_sweep_down = prev['liquidity_sweep_down']
                is_bullish_imbalance = (curr['close'] > curr['open'] and 
                                      curr['body_ratio'] > 0.5)
                wick_rejection = (curr['low'] - curr['open']) > (curr['close'] - curr['open'])
                
                if all([is_liquidity_sweep_down, is_bullish_imbalance, vol_confirm, is_valid_session, wick_rejection, is_not_inside_bar]):
                    entry_price = curr['close']
                    stop_loss = curr['low']
                    target = entry_price + 2 * (entry_price - stop_loss)
                    confidence = 0.7
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=prev['swing_low_price'] if 'swing_low_price' in df.columns else prev['low']
                    )
                    signals.append(signal)
                    
                # Short setup
                is_liquidity_sweep_up = prev['liquidity_sweep_up']
                is_bearish_imbalance = (curr['close'] < curr['open'] and 
                                      curr['body_ratio'] > 0.5)
                wick_rejection_short = (curr['high'] - curr['open']) > (curr['open'] - curr['close'])
                
                if all([is_liquidity_sweep_up, is_bearish_imbalance, vol_confirm, is_valid_session, wick_rejection_short, is_not_inside_bar]):
                    entry_price = curr['close']
                    stop_loss = curr['high']
                    target = entry_price - 2 * (stop_loss - entry_price)
                    confidence = 0.7
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=prev['swing_high_price'] if 'swing_high_price' in df.columns else prev['high']
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error in Body Imbalance strategy: {e}")
            
        return signals

class OrderBlockFVGWrapper:
    """Wrapper for Order Block + FVG strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "order_block"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run strategy on provided data"""
        signals = []
        
        try:
            if len(df) < 50:
                return signals
                
            df = df.copy()
            
            # Order block detection
            ob_lookback = 3
            min_body_ratio = 0.3
            
            df['bullish_ob'] = False
            df['bearish_ob'] = False
            
            for i in range(ob_lookback, len(df)):
                # Bullish order block
                if (not df.iloc[i-1]['is_bullish'] and 
                    df.iloc[i]['is_bullish'] and
                    df.iloc[i-1]['body_ratio'] > min_body_ratio):
                    
                    lookback_high = max(df.loc[i-ob_lookback:i-1]['high'])
                    if df.iloc[i]['close'] > lookback_high:
                        df.loc[i-1, 'bullish_ob'] = True
                        
                # Bearish order block
                if (df.iloc[i-1]['is_bullish'] and 
                    not df.iloc[i]['is_bullish'] and
                    df.iloc[i-1]['body_ratio'] > min_body_ratio):
                    
                    lookback_low = min(df.loc[i-ob_lookback:i-1]['low'])
                    if df.iloc[i]['close'] < lookback_low:
                        df.loc[i-1, 'bearish_ob'] = True
                        
            # FVG detection
            df['bullish_fvg'] = False
            df['bearish_fvg'] = False
            
            for i in range(2, len(df)):
                if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                    df.loc[i, 'bullish_fvg'] = True
                if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                    df.loc[i, 'bearish_fvg'] = True
                    
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=5).mean()
            
            # Generate signals
            for i in range(len(df)):
                if pd.isna(df.iloc[i]['atr']):
                    continue
                    
                # Buy signal
                if df.iloc[i]['bullish_ob']:
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['low'] - df.iloc[i]['atr']
                    target = entry_price + 1.5 * df.iloc[i]['atr']
                    confidence = 0.6
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i]['low']
                    )
                    signals.append(signal)
                    
                # Sell signal
                elif df.iloc[i]['bearish_ob']:
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['high'] + df.iloc[i]['atr']
                    target = entry_price - 1.5 * df.iloc[i]['atr']
                    confidence = 0.6
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i]['high']
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error in Order Block strategy: {e}")
            
        return signals

class StockBurnerWrapper:
    """Wrapper for Stock Burner strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "stock_burner"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run strategy on provided data"""
        signals = []
        
        try:
            if len(df) < 50:
                return signals
                
            df = df.copy()
            
            # EMA calculations
            ema_fast = 9
            ema_slow = 20
            
            df[f'ema_{ema_fast}'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
            df[f'ema_{ema_slow}'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
            
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Trend strength filter
            gap_atr_mult = 0.10
            slope_lookback = 3
            min_slope = 0
            
            ema_f = df[f'ema_{ema_fast}']
            ema_s = df[f'ema_{ema_slow}']
            atr = df['atr'].fillna(method='bfill').fillna(method='ffill')
            
            gap_ok = (ema_f - ema_s).abs() > (gap_atr_mult * atr)
            slope_f = ema_f - ema_f.shift(slope_lookback)
            slope_s = ema_s - ema_s.shift(slope_lookback)
            slope_ok = (slope_f * slope_s > 0)
            strong_trend = gap_ok & slope_ok
            
            # Generate signals
            for i in range(len(df)):
                if not strong_trend.iloc[i]:
                    continue
                    
                ef = float(ema_f.iloc[i])
                es = float(ema_s.iloc[i])
                high = float(df.iloc[i]['high'])
                low = float(df.iloc[i]['low'])
                close = float(df.iloc[i]['close'])
                
                # Buy signal (pullback to EMA9 during strong uptrend)
                if ef > es and (low <= ef <= high):
                    entry_price = np.clip(ef, low, high)
                    stop_loss = low
                    risk = max(entry_price - stop_loss, 0.01)
                    target = entry_price + 3 * risk  # RR = 3.0
                    confidence = 0.5
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=ef
                    )
                    signals.append(signal)
                    
                # Sell signal (pullback to EMA9 during strong downtrend)
                elif ef < es and (low <= ef <= high):
                    entry_price = np.clip(ef, low, high)
                    stop_loss = high
                    risk = max(stop_loss - entry_price, 0.01)
                    target = entry_price - 3 * risk  # RR = 3.0
                    confidence = 0.5
                    
                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=ef
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error in Stock Burner strategy: {e}")
            
        return signals

class LiveStrategyRunner:
    """Main strategy runner that coordinates all strategies"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize strategy wrappers
        self.strategies = {
            'vol_spike': VolumeSpikeLiquiditySweepWrapper(self.config.get('vol_spike', {})),
            'body_imbalance': BodyImbalanceWrapper(self.config.get('body_imbalance', {})),
            'order_block': OrderBlockFVGWrapper(self.config.get('order_block', {})),
            'stock_burner': StockBurnerWrapper(self.config.get('stock_burner', {}))
        }
        
        # Strategy weights
        self.strategy_weights = self.config.get('strategy_weights', {
            'vol_spike': 0.35,
            'body_imbalance': 0.25,
            'order_block': 0.25,
            'stock_burner': 0.15
        })
        
        # Timeframe weights
        self.timeframe_weights = self.config.get('timeframe_weights', {
            '3m': 0.5, '5m': 0.6, '15m': 0.7,
            '60m': 1.0, '120m': 1.2, '180m': 1.3,
            '240m': 1.4, '1D': 1.6
        })
        
        # Signal cache
        self.signal_cache = {}
        self.max_cache_size = 1000
        
    def run_all_strategies(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run all enabled strategies on the provided data"""
        all_signals = []
        
        try:
            # Add required columns if missing
            df = self._ensure_required_columns(df)
            
            # Run each strategy
            for strategy_name, strategy in self.strategies.items():
                try:
                    signals = strategy.run_strategy(df, symbol, timeframe)
                    
                    # Apply weights to signals
                    for signal in signals:
                        signal.metadata['strategy_weight'] = self.strategy_weights.get(strategy_name, 0.25)
                        signal.metadata['timeframe_weight'] = self.timeframe_weights.get(timeframe, 1.0)
                        
                    all_signals.extend(signals)
                    logger.debug(f"{strategy_name} generated {len(signals)} signals for {symbol} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error running {strategy_name} strategy: {e}")
                    
        except Exception as e:
            logger.error(f"Error running strategies for {symbol} {timeframe}: {e}")
            
        return all_signals
        
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has all required columns"""
        df = df.copy()
        
        # Basic OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'date']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                
        # Add derived columns if missing
        if 'is_bullish' not in df.columns:
            df['is_bullish'] = df['close'] > df['open']
            
        if 'body' not in df.columns:
            df['body'] = abs(df['close'] - df['open'])
            
        if 'upper_wick' not in df.columns:
            df['upper_wick'] = df[['open', 'close']].max(axis=1) - df['high']
            
        if 'lower_wick' not in df.columns:
            df['lower_wick'] = df['low'] - df[['open', 'close']].min(axis=1)
            
        if 'body_ratio' not in df.columns:
            df['range'] = df['high'] - df['low']
            df['body_ratio'] = df['body'] / (df['range'] + 1e-9)
            
        return df
        
    def get_strategy_summary(self) -> Dict:
        """Get summary of configured strategies"""
        return {
            'strategies': list(self.strategies.keys()),
            'strategy_weights': self.strategy_weights,
            'timeframe_weights': self.timeframe_weights,
            'total_strategies': len(self.strategies)
        }
