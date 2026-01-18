"""
Live Signal Aggregator Module
Aggregates signals from multiple strategies, applies weighting logic, and produces final actionable decisions
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import json

from live_strategy_runner import StrategySignal

logger = logging.getLogger(__name__)

@dataclass
class AggregatedSignal:
    """Final aggregated signal with complete metadata"""
    symbol: str
    action: str  # BUY, SELL, NEUTRAL
    confidence: float
    entry_price: float
    stop_loss_level: float
    target_level: float
    signal_timestamp: datetime
    signal_timeframe: str
    chart_anchor_price: float
    timestamp_generated: datetime
    aggregated_confidence: float
    final_action: str
    contributing_strategies: List[str]
    contributing_timeframes: List[str]
    decision_score: float
    version_id: str
    raw_strategy_outputs: List[Dict]
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss_level': self.stop_loss_level,
            'target_level': self.target_level,
            'signal_timestamp': self.signal_timestamp.isoformat(),
            'signal_timeframe': self.signal_timeframe,
            'chart_anchor_price': self.chart_anchor_price,
            'timestamp_generated': self.timestamp_generated.isoformat(),
            'aggregated_confidence': self.aggregated_confidence,
            'final_action': self.final_action,
            'contributing_strategies': self.contributing_strategies,
            'contributing_timeframes': self.contributing_timeframes,
            'decision_score': self.decision_score,
            'version_id': self.version_id,
            'raw_strategy_outputs': self.raw_strategy_outputs,
            'metadata': self.metadata or {}
        }

class SignalAggregator:
    """Aggregates signals from multiple strategies using weighted logic"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
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
        
        # Aggregation settings
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.3)
        self.confluence_threshold = self.config.get('confluence_threshold', 2)  # Minimum strategies to agree
        self.max_signals_per_symbol = self.config.get('max_signals_per_symbol', 3)
        
        # Version tracking
        self.version_id = self.config.get('version_id', 'v1.0.0')
        
        # Signal history for deduplication
        self.recent_signals = defaultdict(list)
        self.signal_dedup_window = timedelta(minutes=5)
        
    def aggregate_signals(self, signals: List[StrategySignal], symbol: str) -> List[AggregatedSignal]:
        """Aggregate signals for a symbol"""
        if not signals:
            return []
            
        try:
            # Group signals by action (BUY/SELL)
            buy_signals = [s for s in signals if s.action == 'BUY']
            sell_signals = [s for s in signals if s.action == 'SELL']
            
            aggregated_signals = []
            
            # Process buy signals
            if len(buy_signals) >= self.confluence_threshold:
                agg_signal = self._create_aggregated_signal(buy_signals, 'BUY', symbol)
                if agg_signal and self._is_valid_signal(agg_signal):
                    aggregated_signals.append(agg_signal)
                    
            # Process sell signals
            if len(sell_signals) >= self.confluence_threshold:
                agg_signal = self._create_aggregated_signal(sell_signals, 'SELL', symbol)
                if agg_signal and self._is_valid_signal(agg_signal):
                    aggregated_signals.append(agg_signal)
                    
            # Sort by confidence and limit signals
            aggregated_signals.sort(key=lambda x: x.decision_score, reverse=True)
            return aggregated_signals[:self.max_signals_per_symbol]
            
        except Exception as e:
            logger.error(f"Error aggregating signals for {symbol}: {e}")
            return []
            
    def _create_aggregated_signal(self, signals: List[StrategySignal], action: str, symbol: str) -> Optional[AggregatedSignal]:
        """Create aggregated signal from multiple signals"""
        try:
            if not signals:
                return None
                
            # Get unique strategies and timeframes
            strategies = list(set(s.strategy_name for s in signals))
            timeframes = list(set(s.timeframe for s in signals))
            
            # Calculate weighted confidence
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for signal in signals:
                strategy_weight = self.strategy_weights.get(signal.strategy_name, 0.25)
                timeframe_weight = self.timeframe_weights.get(signal.timeframe, 1.0)
                combined_weight = strategy_weight * timeframe_weight
                
                weighted_confidence += signal.confidence * combined_weight
                total_weight += combined_weight
                
            if total_weight == 0:
                return None
                
            aggregated_confidence = weighted_confidence / total_weight
            
            # Calculate decision score
            strategy_confluence = len(strategies)
            timeframe_confluence = len(timeframes)
            decision_score = (aggregated_confidence * 0.5 + 
                            (strategy_confluence / 4) * 0.3 +  # Normalize by max 4 strategies
                            (timeframe_confluence / 8) * 0.2)  # Normalize by max 8 timeframes
            
            # Determine final action
            if aggregated_confidence >= self.min_confidence_threshold:
                final_action = action
            else:
                final_action = 'NEUTRAL'
                
            # Calculate consensus entry, SL, and target
            entry_prices = [s.entry_price for s in signals]
            stop_losses = [s.stop_loss for s in signals]
            targets = [s.target for s in signals]
            
            consensus_entry = np.median(entry_prices)
            consensus_sl = np.median(stop_losses)
            consensus_target = np.median(targets)
            
            # Get most recent timestamp
            latest_timestamp = max(s.timestamp for s in signals)
            
            # Get anchor price (prefer the most recent signal's anchor)
            anchor_price = signals[-1].anchor_price if signals[-1].anchor_price else consensus_entry
            
            # Prepare raw strategy outputs
            raw_outputs = [s.to_dict() for s in signals]
            
            # Create aggregated signal
            aggregated_signal = AggregatedSignal(
                symbol=symbol,
                action=action,
                confidence=aggregated_confidence,
                entry_price=consensus_entry,
                stop_loss_level=consensus_sl,
                target_level=consensus_target,
                signal_timestamp=latest_timestamp,
                signal_timeframe=self._get_primary_timeframe(signals),
                chart_anchor_price=anchor_price,
                timestamp_generated=datetime.now(),
                aggregated_confidence=aggregated_confidence,
                final_action=final_action,
                contributing_strategies=strategies,
                contributing_timeframes=timeframes,
                decision_score=decision_score,
                version_id=self.version_id,
                raw_strategy_outputs=raw_outputs,
                metadata={
                    'signal_count': len(signals),
                    'strategy_distribution': Counter(s.strategy_name for s in signals),
                    'timeframe_distribution': Counter(s.timeframe for s in signals),
                    'weight_distribution': {s.strategy_name: self.strategy_weights.get(s.strategy_name, 0.25) for s in signals}
                }
            )
            
            return aggregated_signal
            
        except Exception as e:
            logger.error(f"Error creating aggregated signal: {e}")
            return None
            
    def _get_primary_timeframe(self, signals: List[StrategySignal]) -> str:
        """Determine primary timeframe based on weights"""
        timeframe_scores = defaultdict(float)
        
        for signal in signals:
            weight = self.timeframe_weights.get(signal.timeframe, 1.0)
            timeframe_scores[signal.timeframe] += weight
            
        if not timeframe_scores:
            return 'unknown'
            
        return max(timeframe_scores.items(), key=lambda x: x[1])[0]
        
    def _is_valid_signal(self, signal: AggregatedSignal) -> bool:
        """Validate signal quality"""
        try:
            # Check confidence threshold
            if signal.aggregated_confidence < self.min_confidence_threshold:
                return False
                
            # Check for reasonable price levels
            if signal.entry_price <= 0 or signal.stop_loss_level <= 0 or signal.target_level <= 0:
                return False
                
            # Check risk/reward ratio
            if signal.action == 'BUY':
                risk = signal.entry_price - signal.stop_loss_level
                reward = signal.target_level - signal.entry_price
                rr_ratio = reward / risk if risk > 0 else 0
            elif signal.action == 'SELL':
                risk = signal.stop_loss_level - signal.entry_price
                reward = signal.entry_price - signal.target_level
                rr_ratio = reward / risk if risk > 0 else 0
            else:
                rr_ratio = 0
                
            if rr_ratio < 0.5:  # Minimum RR ratio
                return False
                
            # Check for duplicates
            if self._is_duplicate_signal(signal):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
            
    def _is_duplicate_signal(self, signal: AggregatedSignal) -> bool:
        """Check if signal is duplicate of recent signals"""
        try:
            recent_signals = self.recent_signals[signal.symbol]
            current_time = datetime.now()
            
            # Clean old signals
            self.recent_signals[signal.symbol] = [
                s for s in recent_signals 
                if current_time - s['timestamp'] < self.signal_dedup_window
            ]
            
            # Check for similar recent signals
            for recent in self.recent_signals[signal.symbol]:
                if (recent['action'] == signal.action and
                    abs(recent['entry_price'] - signal.entry_price) / signal.entry_price < 0.01):  # 1% tolerance
                    return True
                    
            # Add current signal to recent
            self.recent_signals[signal.symbol].append({
                'action': signal.action,
                'entry_price': signal.entry_price,
                'timestamp': current_time
            })
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate signal: {e}")
            return False
            
    def get_signal_summary(self, signals: List[AggregatedSignal]) -> Dict:
        """Get summary of aggregated signals"""
        if not signals:
            return {'total_signals': 0}
            
        summary = {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.action == 'BUY']),
            'sell_signals': len([s for s in signals if s.action == 'SELL']),
            'neutral_signals': len([s for s in signals if s.action == 'NEUTRAL']),
            'average_confidence': np.mean([s.aggregated_confidence for s in signals]),
            'average_decision_score': np.mean([s.decision_score for s in signals]),
            'strategies_used': list(set(s for signal in signals for s in signal.contributing_strategies)),
            'timeframes_used': list(set(s for signal in signals for s in signal.contributing_timeframes)),
            'symbols': list(set(s.symbol for s in signals))
        }
        
        return summary
        
    def filter_signals_by_confidence(self, signals: List[AggregatedSignal], min_confidence: float) -> List[AggregatedSignal]:
        """Filter signals by minimum confidence"""
        return [s for s in signals if s.aggregated_confidence >= min_confidence]
        
    def filter_signals_by_symbol(self, signals: List[AggregatedSignal], symbols: List[str]) -> List[AggregatedSignal]:
        """Filter signals by specific symbols"""
        return [s for s in signals if s.symbol in symbols]
        
    def get_top_signals(self, signals: List[AggregatedSignal], count: int = 5) -> List[AggregatedSignal]:
        """Get top N signals by decision score"""
        return sorted(signals, key=lambda x: x.decision_score, reverse=True)[:count]

class SignalConsolidator:
    """Consolidates signals across multiple symbols and timeframes"""
    
    def __init__(self, aggregator: SignalAggregator):
        self.aggregator = aggregator
        self.consolidated_signals = []
        self.max_consolidated_signals = 100
        
    def consolidate_all_signals(self, symbol_signals: Dict[str, List[StrategySignal]]) -> List[AggregatedSignal]:
        """Consolidate signals across all symbols"""
        all_aggregated = []
        
        for symbol, signals in symbol_signals.items():
            aggregated = self.aggregator.aggregate_signals(signals, symbol)
            all_aggregated.extend(aggregated)
            
        # Sort by decision score
        all_aggregated.sort(key=lambda x: x.decision_score, reverse=True)
        
        # Keep only top signals
        self.consolidated_signals = all_aggregated[:self.max_consolidated_signals]
        
        return self.consolidated_signals
        
    def get_market_overview(self) -> Dict:
        """Get market overview from consolidated signals"""
        if not self.consolidated_signals:
            return {'status': 'no_signals'}
            
        buy_signals = [s for s in self.consolidated_signals if s.final_action == 'BUY']
        sell_signals = [s for s in self.consolidated_signals if s.final_action == 'SELL']
        
        overview = {
            'total_signals': len(self.consolidated_signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'market_bias': 'bullish' if len(buy_signals) > len(sell_signals) else 'bearish',
            'average_confidence': np.mean([s.aggregated_confidence for s in self.consolidated_signals]),
            'top_symbols': list(set(s.symbol for s in self.consolidated_signals[:10])),
            'top_strategies': list(set(s for signal in self.consolidated_signals for s in signal.contributing_strategies)),
            'timestamp': datetime.now().isoformat()
        }
        
        return overview
        
    def export_signals_json(self, filepath: str) -> bool:
        """Export consolidated signals to JSON file"""
        try:
            signals_data = [signal.to_dict() for signal in self.consolidated_signals]
            
            with open(filepath, 'w') as f:
                json.dump(signals_data, f, indent=2, default=str)
                
            logger.info(f"Exported {len(signals_data)} signals to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting signals to JSON: {e}")
            return False
            
    def export_signals_csv(self, filepath: str) -> bool:
        """Export consolidated signals to CSV file"""
        try:
            if not self.consolidated_signals:
                return False
                
            # Convert to DataFrame
            data = []
            for signal in self.consolidated_signals:
                row = signal.to_dict()
                row['contributing_strategies'] = ','.join(row['contributing_strategies'])
                row['contributing_timeframes'] = ','.join(row['contributing_timeframes'])
                data.append(row)
                
            df = pd.DataFrame(data)
            
            # Select key columns for CSV
            csv_columns = [
                'symbol', 'final_action', 'aggregated_confidence', 'decision_score',
                'entry_price', 'stop_loss_level', 'target_level', 'signal_timeframe',
                'contributing_strategies', 'contributing_timeframes', 'chart_anchor_price',
                'timestamp_generated', 'version_id'
            ]
            
            df[csv_columns].to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} signals to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting signals to CSV: {e}")
            return False
