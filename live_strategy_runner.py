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

import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


def _tf_minutes(tf: str) -> int:
    t = str(tf or "").strip().lower()
    if t.endswith("m"):
        try:
            return int(t[:-1])
        except Exception:
            return 0
    if t.endswith("h"):
        try:
            return int(t[:-1]) * 60
        except Exception:
            return 0
    if t in {"1d", "d", "day", "1day"}:
        return 60 * 24
    try:
        return int(t)
    except Exception:
        return 0


def _find_recent_swings(df: pd.DataFrame, lookback: int = 3) -> Tuple[Optional[float], Optional[float]]:
    if df is None or df.empty or len(df) < (lookback * 2 + 3):
        return None, None
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    for i in range(lookback, len(df) - lookback):
        h = float(highs[i])
        l = float(lows[i])
        is_sh = True
        is_sl = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if float(highs[j]) >= h:
                is_sh = False
            if float(lows[j]) <= l:
                is_sl = False
            if not is_sh and not is_sl:
                break
        if is_sh:
            swing_highs.append(h)
        if is_sl:
            swing_lows.append(l)
    return (swing_highs[-1] if swing_highs else None), (swing_lows[-1] if swing_lows else None)


def _equal_level(values: List[float], rel_tol: float = 0.0009) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    v_sorted = sorted([float(v) for v in values if v is not None])
    if len(v_sorted) < 2:
        return None
    clusters: List[List[float]] = []
    for v in v_sorted:
        placed = False
        for c in clusters:
            ref = float(sum(c) / max(len(c), 1))
            if ref > 0 and abs(v - ref) / ref <= rel_tol:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    clusters = [c for c in clusters if len(c) >= 2]
    if not clusters:
        return None
    best = max(clusters, key=lambda c: len(c))
    return float(sum(best) / len(best))


def _resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df is None or df.empty or minutes <= 0:
        return pd.DataFrame()
    out = df.copy()
    if "date" not in out.columns:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out = out.set_index("date")
    rule = f"{int(minutes)}min"
    o = out["open"].resample(rule).first()
    h = out["high"].resample(rule).max()
    l = out["low"].resample(rule).min()
    c = out["close"].resample(rule).last()
    v = out["volume"].resample(rule).sum() if "volume" in out.columns else None
    res = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    if v is not None:
        res["volume"] = v
    res = res.dropna().reset_index()
    return res

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
                 metadata: Dict = None,
                 timestamp: Any = None):
        
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
        if timestamp is None:
            self.timestamp = datetime.now(IST)
        else:
            try:
                parsed = pd.to_datetime(timestamp)
                if getattr(parsed, "tzinfo", None) is None:
                    parsed = parsed.tz_localize(IST)  # type: ignore[union-attr]
                else:
                    parsed = parsed.tz_convert(IST)  # type: ignore[union-attr]
                self.timestamp = parsed.to_pydatetime()  # type: ignore[union-attr]
            except Exception:
                self.timestamp = datetime.now(IST)
        
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
            'timestamp': self.timestamp.strftime('%d-%b-%Y %I:%M %p')
        }

class VolumeSpikeLiquiditySweepWrapper:
    """Wrapper for Volume Spike + Liquidity Sweep strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "vol_spike"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run strategy on provided data"""
        signals: List[StrategySignal] = []

        try:
            if len(df) < 50:  # Need minimum data
                return signals

            df = df.copy()

            # Volume spike detection
            volume_lookback = self.config.get('volume_lookback', 20)
            spike_threshold = self.config.get('spike_threshold', 2.0)

            df['volume_ma'] = df['volume'].rolling(window=volume_lookback).mean()
            df['volume_spike'] = df['volume'] > (df['volume_ma'] * spike_threshold)
            df['volume_ratio'] = df['volume'] / df['volume_ma']  # ADD THIS
            df['is_bullish'] = df['close'] > df['open']  # ADD THIS

            # Swing point detection (simplified)
            swing_lookback = 5
            df['swing_high'] = False
            df['swing_low'] = False

            for i in range(swing_lookback, len(df) - swing_lookback):
                current_high = df.iloc[i]['high']
                current_low = df.iloc[i]['low']

                # Check swing high
                is_swing_high = all(
                    df.iloc[j]['high'] < current_high
                    for j in range(i - swing_lookback, i + swing_lookback + 1)
                    if j != i
                )
                if is_swing_high:
                    df.loc[i, 'swing_high'] = True

                # Check swing low
                is_swing_low = all(
                    df.iloc[j]['low'] > current_low
                    for j in range(i - swing_lookback, i + swing_lookback + 1)
                    if j != i
                )
                if is_swing_low:
                    df.loc[i, 'swing_low'] = True

            # Liquidity sweep detection
            df['liquidity_sweep_down'] = False
            df['liquidity_sweep_up'] = False

            for i in range(1, len(df)):
                if df.iloc[i - 1]['swing_low'] and df.iloc[i]['low'] < df.iloc[i - 1]['low']:
                    df.loc[i, 'liquidity_sweep_down'] = True

                if df.iloc[i - 1]['swing_high'] and df.iloc[i]['high'] > df.iloc[i - 1]['high']:
                    df.loc[i, 'liquidity_sweep_up'] = True

            # Generate signals
            for i in range(2, len(df)):
                ts = df["date"].iloc[i] if "date" in df.columns else None
                if (
                    df.iloc[i]['volume_spike']
                    and df.iloc[i - 1]['liquidity_sweep_down']
                    and df.iloc[i]['is_bullish']
                ):
                    # Buy signal
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['low']
                    target = entry_price + 3 * (entry_price - stop_loss)
                    confidence = min(0.8, df.iloc[i].get('volume_ratio', 1.0) / 3.0)

                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i - 1].get('swing_low_price', df.iloc[i - 1]['low']),
                        timestamp=ts,
                    )
                    signals.append(signal)

                elif (
                    df.iloc[i]['volume_spike']
                    and df.iloc[i - 1]['liquidity_sweep_up']
                    and (not df.iloc[i]['is_bullish'])
                ):
                    # Sell signal
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['high']
                    target = entry_price - 3 * (stop_loss - entry_price)
                    confidence = min(0.8, df.iloc[i].get('volume_ratio', 1.0) / 3.0)

                    signal = StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=target,
                        confidence=confidence,
                        anchor_price=df.iloc[i - 1].get('swing_high_price', df.iloc[i - 1]['high']),
                        timestamp=ts,
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
            
            # Compute required columns if missing
            if 'body' not in df.columns:
                df['body'] = abs(df['close'] - df['open'])
            if 'upper_wick' not in df.columns:
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            if 'lower_wick' not in df.columns:
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            
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
                ts = df["date"].iloc[i] if "date" in df.columns else None
                
                # Common filters
                vol_confirm = curr.get('volume_spike', True)
                is_valid_session = curr.get('session', 'morning') in ['morning', 'afternoon']
                is_not_inside_bar = not ((curr['high'] < prev['high']) and (curr['low'] > prev['low']))
                
                # Long setup
                is_liquidity_sweep_down = prev['liquidity_sweep_down']
                is_bullish_imbalance = (curr['close'] > curr['open'] and 
                                      curr['body_ratio'] > 0.5)
                # Fixed: proper wick rejection calculation
                lower_wick_size = curr['open'] - curr['low'] if curr['close'] > curr['open'] else curr['close'] - curr['low']
                body_size = abs(curr['close'] - curr['open'])
                wick_rejection = lower_wick_size > body_size * 0.5
                
                if all([is_liquidity_sweep_down, is_bullish_imbalance, vol_confirm, is_valid_session, wick_rejection, is_not_inside_bar]):
                    entry_price = curr['close']
                    stop_loss = curr['low']
                    target = entry_price + 3 * (entry_price - stop_loss)
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
                        anchor_price=prev['swing_low_price'] if 'swing_low_price' in df.columns else prev['low'],
                        timestamp=ts,
                    )
                    signals.append(signal)
                    
                # Short setup
                is_liquidity_sweep_up = prev['liquidity_sweep_up']
                is_bearish_imbalance = (curr['close'] < curr['open'] and 
                                      curr['body_ratio'] > 0.5)
                # Fixed: proper upper wick rejection calculation
                upper_wick_size = curr['high'] - curr['close'] if curr['close'] < curr['open'] else curr['high'] - curr['open']
                body_size = abs(curr['close'] - curr['open'])
                wick_rejection_short = upper_wick_size > body_size * 0.5
                
                if all([is_liquidity_sweep_up, is_bearish_imbalance, vol_confirm, is_valid_session, wick_rejection_short, is_not_inside_bar]):
                    entry_price = curr['close']
                    stop_loss = curr['high']
                    target = entry_price - 3 * (stop_loss - entry_price)
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
                        anchor_price=prev['swing_high_price'] if 'swing_high_price' in df.columns else prev['high'],
                        timestamp=ts,
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
            
            # Ensure required columns exist
            if 'is_bullish' not in df.columns:
                df['is_bullish'] = df['close'] > df['open']
            if 'body' not in df.columns:
                df['body'] = abs(df['close'] - df['open'])
            if 'upper_wick' not in df.columns:
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            if 'lower_wick' not in df.columns:
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            if 'body_ratio' not in df.columns:
                df['body_ratio'] = df['body'] / (df['upper_wick'] + df['lower_wick'] + df['body'] + 1e-9)
            
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

                ts = df["date"].iloc[i] if "date" in df.columns else None
                    
                # Buy signal
                if df.iloc[i]['bullish_ob']:
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['low'] - df.iloc[i]['atr']
                    target = entry_price + 3 * df.iloc[i]['atr']
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
                        anchor_price=df.iloc[i]['low'],
                        timestamp=ts,
                    )
                    signals.append(signal)
                    
                # Sell signal
                elif df.iloc[i]['bearish_ob']:
                    entry_price = df.iloc[i]['close']
                    stop_loss = df.iloc[i]['high'] + df.iloc[i]['atr']
                    target = entry_price - 3 * df.iloc[i]['atr']
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
                        anchor_price=df.iloc[i]['high'],
                        timestamp=ts,
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

                ts = df["date"].iloc[i] if "date" in df.columns else None
                    
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
                        anchor_price=ef,
                        timestamp=ts,
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
                        anchor_price=ef,
                        timestamp=ts,
                    )
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Error in Stock Burner strategy: {e}")
            
        return signals


class EmaCrossover5_20Wrapper:
    """Wrapper for a simple 5/20 EMA crossover strategy (diagnostic backtest)."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "ema_crossover_5_20"

    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        signals: List[StrategySignal] = []

        try:
            if len(df) < 25:
                return signals

            df = df.copy()

            df["ema_fast_5"] = df["close"].ewm(span=5, adjust=False).mean()
            df["ema_slow_20"] = df["close"].ewm(span=20, adjust=False).mean()

            cross_up = (df["ema_fast_5"] > df["ema_slow_20"]) & (
                df["ema_fast_5"].shift(1) <= df["ema_slow_20"].shift(1)
            )
            cross_down = (df["ema_fast_5"] < df["ema_slow_20"]) & (
                df["ema_fast_5"].shift(1) >= df["ema_slow_20"].shift(1)
            )

            rng = (df["high"] - df["low"]).rolling(window=14).mean()
            rng = rng.fillna(method="bfill").fillna(method="ffill")

            for i in range(len(df)):
                r = float(rng.iloc[i]) if float(rng.iloc[i]) > 0 else 0.01

                ts = df["date"].iloc[i] if "date" in df.columns else None

                if bool(cross_up.iloc[i]):
                    entry = float(df["close"].iloc[i])
                    signals.append(
                        StrategySignal(
                            strategy_name=self.strategy_name,
                            symbol=symbol,
                            timeframe=timeframe,
                            action="BUY",
                            entry_price=entry,
                            stop_loss=entry - 1.0 * r,
                            target=entry + 3.0 * r,
                            confidence=0.4,
                            anchor_price=float(df["ema_slow_20"].iloc[i]),
                            timestamp=ts,
                        )
                    )
                elif bool(cross_down.iloc[i]):
                    entry = float(df["close"].iloc[i])
                    signals.append(
                        StrategySignal(
                            strategy_name=self.strategy_name,
                            symbol=symbol,
                            timeframe=timeframe,
                            action="SELL",
                            entry_price=entry,
                            stop_loss=entry + 1.0 * r,
                            target=entry - 3.0 * r,
                            confidence=0.4,
                            anchor_price=float(df["ema_slow_20"].iloc[i]),
                            timestamp=ts,
                        )
                    )

        except Exception as e:
            logger.error(f"Error in EMA crossover strategy: {e}")

        return signals


class SmcLiquidityTrapWrapper:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "smc_liquidity_trap"

    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        signals: List[StrategySignal] = []

        try:
            if df is None or df.empty or len(df) < 80:
                return signals

            data = df.copy()
            data = data.sort_values("date") if "date" in data.columns else data
            if "is_bullish" not in data.columns:
                data["is_bullish"] = data["close"] > data["open"]

            ltf_min = _tf_minutes(timeframe)
            htf_min = 60 if ltf_min < 60 else max(ltf_min, 60)
            htf = _resample_ohlcv(data[["date", "open", "high", "low", "close", "volume"]], htf_min)
            if htf.empty:
                return signals

            htf_sh, htf_sl = _find_recent_swings(
                htf,
                lookback=int(self.config.get("htf_swing_lookback", 3)),
            )
            if htf_sh is None or htf_sl is None or float(htf_sh) == float(htf_sl):
                return signals

            recent = data.tail(int(self.config.get("ltf_lookback", 140))).copy()
            highs = recent["high"].to_numpy(dtype=float)
            lows = recent["low"].to_numpy(dtype=float)
            closes = recent["close"].to_numpy(dtype=float)

            swing_lb = int(self.config.get("ltf_swing_lookback", 3))
            swing_highs: List[float] = []
            swing_lows: List[float] = []
            for i in range(swing_lb, len(recent) - swing_lb):
                h = float(highs[i])
                l = float(lows[i])
                if all(float(highs[j]) < h for j in range(i - swing_lb, i + swing_lb + 1) if j != i):
                    swing_highs.append(h)
                if all(float(lows[j]) > l for j in range(i - swing_lb, i + swing_lb + 1) if j != i):
                    swing_lows.append(l)

            eq_high = _equal_level(
                swing_highs[-12:],
                rel_tol=float(self.config.get("equal_tol", 0.0009)),
            )
            eq_low = _equal_level(
                swing_lows[-12:],
                rel_tol=float(self.config.get("equal_tol", 0.0009)),
            )

            if len(recent) < 3:
                return signals
            i = len(recent) - 1
            cl0 = float(closes[i])
            cl1 = float(closes[i - 1])
            hi0 = float(highs[i])
            hi1 = float(highs[i - 1])
            lo0 = float(lows[i])
            lo1 = float(lows[i - 1])
            ts = recent["date"].iloc[i] if "date" in recent.columns else None

            htf_low = float(min(htf_sl, htf_sh))
            htf_high = float(max(htf_sl, htf_sh))
            fib50_buy = htf_low + 0.5 * (htf_high - htf_low)
            fib50_sell = htf_high - 0.5 * (htf_high - htf_low)

            sl_buffer = float(self.config.get("sl_buffer", 0.001))
            rr = float(self.config.get("rr", 3.0))

            swept_eq_low = (eq_low is not None) and ((lo1 < float(eq_low)) or (lo0 < float(eq_low)))
            reclaim_eq_low = (eq_low is not None) and ((cl0 > float(eq_low)) or (cl1 > float(eq_low)))
            swept_eq_high = (eq_high is not None) and ((hi1 > float(eq_high)) or (hi0 > float(eq_high)))
            reclaim_eq_high = (eq_high is not None) and ((cl0 < float(eq_high)) or (cl1 < float(eq_high)))

            if swept_eq_low and reclaim_eq_low and cl0 <= fib50_buy:
                entry = cl0
                sl = min(lo0, lo1) * (1.0 - sl_buffer)
                tgt = float(eq_high) if eq_high is not None else float(htf_sh)
                if tgt <= entry:
                    tgt = entry + rr * max(entry - sl, 0.01)
                signals.append(
                    StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="BUY",
                        entry_price=float(entry),
                        stop_loss=float(sl),
                        target=float(tgt),
                        confidence=float(self.config.get("confidence", 0.65)),
                        anchor_price=float(eq_low) if eq_low is not None else float(min(lo0, lo1)),
                        timestamp=ts,
                    )
                )

            if swept_eq_high and reclaim_eq_high and cl0 >= fib50_sell:
                entry = cl0
                sl = max(hi0, hi1) * (1.0 + sl_buffer)
                tgt = float(eq_low) if eq_low is not None else float(htf_sl)
                if tgt >= entry:
                    tgt = entry - rr * max(sl - entry, 0.01)
                signals.append(
                    StrategySignal(
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        action="SELL",
                        entry_price=float(entry),
                        stop_loss=float(sl),
                        target=float(tgt),
                        confidence=float(self.config.get("confidence", 0.65)),
                        anchor_price=float(eq_high) if eq_high is not None else float(max(hi0, hi1)),
                        timestamp=ts,
                    )
                )

        except Exception as e:
            logger.error(f"Error in SMC Liquidity Trap strategy: {e}")

        return signals


class Ema5ScalpingWrapper:
    """Wrapper for 5EMA Scalping strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_name = "ema_scalping_5"
        
    def run_strategy(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[StrategySignal]:
        """Run 5EMA Scalping strategy on provided data"""
        signals: List[StrategySignal] = []
        
        try:
            if len(df) < 5:
                return signals
                
            df = df.copy()
            # Ensure EMA 5 is present
            if 'ema_5' not in df.columns:
                df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
                
            # Iterate for backtesting support (efficiently)
            for i in range(2, len(df)):
                alert = df.iloc[i-1]
                trigger = df.iloc[i]
                ts = df["date"].iloc[i] if "date" in df.columns else datetime.now(IST)
                
                # --- SELL LOGIC (SHORT) ---
                # Condition: Entire candle (low) is above 5 EMA
                is_sell_alert = alert['low'] > alert['ema_5']
                # Bearish confirmation: Close < Open OR a long upper wick
                body = abs(alert['close'] - alert['open'])
                upper_wick = alert['high'] - max(alert['open'], alert['close'])
                is_bearish = (alert['close'] < alert['open']) or (upper_wick > body)
                
                if is_sell_alert and is_bearish:
                    # Triggered if current candle breaks alert candle low
                    if trigger['close'] < alert['low']:
                        risk = float(alert['high'] - alert['low'])
                        if risk > 0:
                            signals.append(
                                StrategySignal(
                                    strategy_name=self.strategy_name,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    action="SELL",
                                    entry_price=float(alert['low']),
                                    stop_loss=float(alert['high']),
                                    target=float(alert['low'] - (risk * 3)),
                                    confidence=float(self.config.get("confidence", 0.70)),
                                    anchor_price=float(alert['high']),
                                    timestamp=ts
                                )
                            )
                
                # --- BUY LOGIC (LONG) ---
                # Condition: Entire candle (high) is below 5 EMA
                is_buy_alert = alert['high'] < alert['ema_5']
                # Bullish confirmation: Close > Open OR a long lower wick
                lower_wick = min(alert['open'], alert['close']) - alert['low']
                is_bullish = (alert['close'] > alert['open']) or (lower_wick > body)
                
                if is_buy_alert and is_bullish:
                    # Triggered if current candle breaks alert candle high
                    if trigger['close'] > alert['high']:
                        risk = float(alert['high'] - alert['low'])
                        if risk > 0:
                            signals.append(
                                StrategySignal(
                                    strategy_name=self.strategy_name,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    action="BUY",
                                    entry_price=float(alert['high']),
                                    stop_loss=float(alert['low']),
                                    target=float(alert['high'] + (risk * 3)),
                                    confidence=float(self.config.get("confidence", 0.70)),
                                    anchor_price=float(alert['low']),
                                    timestamp=ts
                                )
                            )
                    
        except Exception as e:
            logger.error(f"Error in 5EMA Scalping strategy: {e}")
            
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
            'stock_burner': StockBurnerWrapper(self.config.get('stock_burner', {})),
            'ema_crossover_5_20': EmaCrossover5_20Wrapper(self.config.get('ema_crossover_5_20', {})),
            'smc_liquidity_trap': SmcLiquidityTrapWrapper(self.config.get('smc_liquidity_trap', {})),
            'ema_scalping_5': Ema5ScalpingWrapper(self.config.get('ema_scalping_5', {})),
        }
        
        # Strategy weights
        self.strategy_weights = self.config.get('strategy_weights', {
            'vol_spike': 0.35,
            'body_imbalance': 0.25,
            'order_block': 0.25,
            'stock_burner': 0.15,
            'ema_crossover_5_20': 0.10,
            'smc_liquidity_trap': 0.25,
            'ema_scalping_5': 0.40,
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
