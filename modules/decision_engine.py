import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Trading decision with confidence and reasoning"""
    action: str  # 'BUY', 'SELL', 'HOLD', 'WAIT'
    confidence: float  # 0-1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    reasoning: List[str]
    signals_supporting: List[str]
    signals_against: List[str]
    urgency: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timeframe: str
    expected_return: float
    max_risk: float

class DecisionEngine:
    """Advanced decision engine for trading recommendations"""
    
    def __init__(self):
        self.signal_weights = {
            'pattern_confirmation': 0.25,
            'technical_indicators': 0.25,
            'momentum': 0.20,
            'volume_confirmation': 0.15,
            'risk_metrics': 0.15
        }
        
    def make_trading_decision(self, symbol: str, data: pd.DataFrame, patterns: List[Dict[str, Any]], 
                            advanced_analytics, account_size: float = 10000) -> TradingDecision:
        """Generate comprehensive trading decision with reasoning"""
        
        current_price = data['Close'].iloc[-1]
        
        # Collect all signals
        signals = self._collect_all_signals(data, patterns, advanced_analytics)
        
        # Calculate weighted decision score
        decision_score = self._calculate_decision_score(signals)
        
        # Determine action based on score
        action = self._determine_action(decision_score)
        
        # Calculate risk metrics
        risk_metrics = advanced_analytics.calculate_risk_metrics(data, 1000)
        
        # Generate entry, stop loss, and take profit levels
        levels = self._calculate_trading_levels(current_price, signals, risk_metrics, action)
        
        # Calculate position size based on risk
        position_size = self._calculate_position_size(account_size, levels['stop_loss'], current_price, action)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(signals, decision_score, action)
        
        # Determine urgency
        urgency = self._determine_urgency(signals, decision_score)
        
        # Calculate expected return and risk
        expected_return = self._calculate_expected_return(levels, current_price, action)
        max_risk = self._calculate_max_risk(levels, current_price, action, position_size, account_size)
        
        return TradingDecision(
            action=action,
            confidence=abs(decision_score),
            entry_price=levels['entry'],
            stop_loss=levels['stop_loss'],
            take_profit=levels['take_profit'],
            position_size=position_size,
            risk_reward_ratio=levels['risk_reward_ratio'],
            reasoning=reasoning['main_reasons'],
            signals_supporting=reasoning['supporting'],
            signals_against=reasoning['against'],
            urgency=urgency,
            timeframe=self._determine_timeframe(signals),
            expected_return=expected_return,
            max_risk=max_risk
        )
    
    def _collect_all_signals(self, data: pd.DataFrame, patterns: List[Dict[str, Any]], 
                           advanced_analytics) -> Dict[str, Any]:
        """Collect all trading signals from different sources"""
        
        # Pattern signals
        pattern_signals = self._analyze_pattern_signals(patterns)
        
        # Technical indicator signals
        tech_signals = self._analyze_technical_signals(data, advanced_analytics)
        
        # Momentum signals
        momentum_signals = self._analyze_momentum_signals(data)
        
        # Volume signals
        volume_signals = self._analyze_volume_signals(data)
        
        # Support/Resistance signals
        sr_signals = self._analyze_support_resistance_signals(data, advanced_analytics)
        
        # Divergence signals
        divergence_signals = advanced_analytics.detect_divergences(data, patterns)
        
        return {
            'patterns': pattern_signals,
            'technical': tech_signals,
            'momentum': momentum_signals,
            'volume': volume_signals,
            'support_resistance': sr_signals,
            'divergences': divergence_signals,
            'overall_trend': self._determine_overall_trend(data)
        }
    
    def _analyze_pattern_signals(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze candlestick pattern signals"""
        
        if not patterns:
            return {'strength': 0, 'direction': 'neutral', 'confidence': 0}
        
        # Get most recent patterns (last 3)
        recent_patterns = patterns[-3:] if len(patterns) >= 3 else patterns
        
        bullish_score = 0
        bearish_score = 0
        total_confidence = 0
        
        for pattern in recent_patterns:
            signal = pattern.get('signal', '')
            confidence = pattern.get('confidence', 0)
            
            if 'Bullish' in signal:
                bullish_score += confidence
            elif 'Bearish' in signal:
                bearish_score += confidence
                
            total_confidence += confidence
        
        net_score = (bullish_score - bearish_score) / len(recent_patterns) if recent_patterns else 0
        avg_confidence = total_confidence / len(recent_patterns) if recent_patterns else 0
        
        return {
            'strength': abs(net_score),
            'direction': 'bullish' if net_score > 0 else 'bearish' if net_score < 0 else 'neutral',
            'confidence': avg_confidence,
            'pattern_count': len(recent_patterns),
            'recent_patterns': [p['type'] for p in recent_patterns]
        }
    
    def _analyze_technical_signals(self, data: pd.DataFrame, advanced_analytics) -> Dict[str, Any]:
        """Analyze technical indicator signals"""
        
        # Calculate indicators
        rsi = advanced_analytics.calculate_rsi(data)
        macd = advanced_analytics.calculate_macd(data)
        bb = advanced_analytics.calculate_bollinger_bands(data)
        stoch = advanced_analytics.calculate_stochastic(data)
        
        current_rsi = rsi.iloc[-1]
        current_macd = macd['macd'].iloc[-1]
        current_signal = macd['signal'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        signals = []
        score = 0
        
        # RSI signals
        if current_rsi < 30:
            signals.append("RSI Oversold - Bullish")
            score += 0.7
        elif current_rsi > 70:
            signals.append("RSI Overbought - Bearish")
            score -= 0.7
        elif 30 <= current_rsi <= 45:
            signals.append("RSI Slightly Oversold - Mildly Bullish")
            score += 0.3
        elif 55 <= current_rsi <= 70:
            signals.append("RSI Slightly Overbought - Mildly Bearish")
            score -= 0.3
        
        # MACD signals
        if current_macd > current_signal and macd['macd'].iloc[-2] <= macd['signal'].iloc[-2]:
            signals.append("MACD Bullish Crossover")
            score += 0.8
        elif current_macd < current_signal and macd['macd'].iloc[-2] >= macd['signal'].iloc[-2]:
            signals.append("MACD Bearish Crossover")
            score -= 0.8
        elif current_macd > current_signal:
            signals.append("MACD Above Signal Line")
            score += 0.3
        else:
            signals.append("MACD Below Signal Line")
            score -= 0.3
        
        # Bollinger Bands signals
        if current_price <= bb['lower'].iloc[-1]:
            signals.append("Price at Lower Bollinger Band - Oversold")
            score += 0.5
        elif current_price >= bb['upper'].iloc[-1]:
            signals.append("Price at Upper Bollinger Band - Overbought")
            score -= 0.5
        
        # Stochastic signals
        current_k = stoch['k'].iloc[-1]
        if current_k < 20:
            signals.append("Stochastic Oversold")
            score += 0.4
        elif current_k > 80:
            signals.append("Stochastic Overbought")
            score -= 0.4
        
        return {
            'score': np.clip(score, -1, 1),
            'signals': signals,
            'rsi': current_rsi,
            'macd_position': 'above' if current_macd > current_signal else 'below',
            'bb_position': self._get_bb_position(current_price, bb)
        }
    
    def _analyze_momentum_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price momentum signals"""
        
        # Calculate price changes
        price_change_1d = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        price_change_5d = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6] if len(data) > 5 else 0
        price_change_20d = (data['Close'].iloc[-1] - data['Close'].iloc[-21]) / data['Close'].iloc[-21] if len(data) > 20 else 0
        
        # Calculate moving average positions
        sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) > 20 else data['Close'].iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else data['Close'].iloc[-1]
        
        current_price = data['Close'].iloc[-1]
        
        signals = []
        score = 0
        
        # Short-term momentum
        if price_change_1d > 0.02:  # 2% gain
            signals.append("Strong Daily Momentum - Bullish")
            score += 0.6
        elif price_change_1d < -0.02:  # 2% loss
            signals.append("Strong Daily Momentum - Bearish")
            score -= 0.6
        
        # Medium-term momentum
        if price_change_5d > 0.05:  # 5% gain in 5 days
            signals.append("Strong Weekly Momentum - Bullish")
            score += 0.4
        elif price_change_5d < -0.05:  # 5% loss in 5 days
            signals.append("Strong Weekly Momentum - Bearish")
            score -= 0.4
        
        # Moving average signals
        if current_price > sma_20 > sma_50:
            signals.append("Price Above Both Moving Averages - Bullish Trend")
            score += 0.5
        elif current_price < sma_20 < sma_50:
            signals.append("Price Below Both Moving Averages - Bearish Trend")
            score -= 0.5
        
        return {
            'score': np.clip(score, -1, 1),
            'signals': signals,
            'daily_change': price_change_1d,
            'weekly_change': price_change_5d,
            'monthly_change': price_change_20d,
            'ma_position': 'bullish' if current_price > sma_20 > sma_50 else 'bearish' if current_price < sma_20 < sma_50 else 'mixed'
        }
    
    def _analyze_volume_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume confirmation signals"""
        
        if 'Volume' not in data.columns:
            return {'score': 0, 'signals': ['Volume data not available']}
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1] if len(data) > 20 else current_volume
        
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
        
        signals = []
        score = 0
        
        # Volume confirmation signals
        if volume_ratio > 2.0:  # Very high volume
            if price_change > 0:
                signals.append("High Volume with Price Increase - Strong Bullish")
                score += 0.8
            else:
                signals.append("High Volume with Price Decrease - Strong Bearish")
                score -= 0.8
        elif volume_ratio > 1.5:  # Above average volume
            if price_change > 0:
                signals.append("Above Average Volume with Price Increase - Bullish")
                score += 0.4
            else:
                signals.append("Above Average Volume with Price Decrease - Bearish")
                score -= 0.4
        elif volume_ratio < 0.5:  # Low volume
            signals.append("Low Volume - Weak Signal")
            score *= 0.5  # Reduce confidence due to low volume
        
        return {
            'score': np.clip(score, -1, 1),
            'signals': signals,
            'volume_ratio': volume_ratio,
            'volume_trend': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.7 else 'normal'
        }
    
    def _analyze_support_resistance_signals(self, data: pd.DataFrame, advanced_analytics) -> Dict[str, Any]:
        """Analyze support and resistance level signals"""
        
        levels = advanced_analytics.calculate_support_resistance_levels(data)
        current_price = data['Close'].iloc[-1]
        
        signals = []
        score = 0
        
        # Check proximity to support/resistance
        support_levels = levels.get('support', [])
        resistance_levels = levels.get('resistance', [])
        
        # Find nearest levels
        nearest_support = max([s for s in support_levels if s < current_price], default=0)
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
        
        support_distance = (current_price - nearest_support) / current_price if nearest_support > 0 else 1
        resistance_distance = (nearest_resistance - current_price) / current_price if nearest_resistance != float('inf') else 1
        
        # Support level signals
        if support_distance < 0.02:  # Within 2% of support
            signals.append("Near Strong Support Level - Potential Bounce")
            score += 0.6
        elif support_distance < 0.05:  # Within 5% of support
            signals.append("Approaching Support Level")
            score += 0.3
        
        # Resistance level signals
        if resistance_distance < 0.02:  # Within 2% of resistance
            signals.append("Near Strong Resistance Level - Potential Rejection")
            score -= 0.6
        elif resistance_distance < 0.05:  # Within 5% of resistance
            signals.append("Approaching Resistance Level")
            score -= 0.3
        
        return {
            'score': np.clip(score, -1, 1),
            'signals': signals,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance
        }
    
    def _calculate_decision_score(self, signals: Dict[str, Any]) -> float:
        """Calculate weighted decision score from all signals"""
        
        score = 0
        
        # Pattern signals
        pattern_score = 0
        if signals['patterns']['direction'] == 'bullish':
            pattern_score = signals['patterns']['strength'] * signals['patterns']['confidence']
        elif signals['patterns']['direction'] == 'bearish':
            pattern_score = -signals['patterns']['strength'] * signals['patterns']['confidence']
        
        score += pattern_score * self.signal_weights['pattern_confirmation']
        
        # Technical indicator signals
        score += signals['technical']['score'] * self.signal_weights['technical_indicators']
        
        # Momentum signals
        score += signals['momentum']['score'] * self.signal_weights['momentum']
        
        # Volume signals
        score += signals['volume']['score'] * self.signal_weights['volume_confirmation']
        
        # Support/Resistance signals
        score += signals['support_resistance']['score'] * self.signal_weights['risk_metrics']
        
        return np.clip(score, -1, 1)
    
    def _determine_action(self, decision_score: float) -> str:
        """Determine trading action based on decision score"""
        
        if decision_score > 0.6:
            return 'BUY'
        elif decision_score > 0.3:
            return 'WEAK_BUY'
        elif decision_score < -0.6:
            return 'SELL'
        elif decision_score < -0.3:
            return 'WEAK_SELL'
        elif abs(decision_score) < 0.1:
            return 'HOLD'
        else:
            return 'WAIT'
    
    def _calculate_trading_levels(self, current_price: float, signals: Dict[str, Any], 
                                risk_metrics: Dict[str, Any], action: str) -> Dict[str, float]:
        """Calculate entry, stop loss, and take profit levels"""
        
        atr = risk_metrics.get('current_atr', current_price * 0.02)
        
        if action in ['BUY', 'WEAK_BUY']:
            entry = current_price
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
            
            # Adjust based on support/resistance
            nearest_support = signals['support_resistance'].get('nearest_support', 0)
            if nearest_support > 0 and nearest_support > stop_loss:
                stop_loss = nearest_support * 0.995  # Just below support
                
        elif action in ['SELL', 'WEAK_SELL']:
            entry = current_price
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
            
            # Adjust based on support/resistance
            nearest_resistance = signals['support_resistance'].get('nearest_resistance', float('inf'))
            if nearest_resistance != float('inf') and nearest_resistance < stop_loss:
                stop_loss = nearest_resistance * 1.005  # Just above resistance
                
        else:  # HOLD or WAIT
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2)
        
        # Calculate risk/reward ratio
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_position_size(self, account_size: float, stop_loss: float, 
                               entry_price: float, action: str) -> float:
        """Calculate optimal position size based on risk"""
        
        risk_per_trade = 0.02  # Risk 2% per trade
        max_loss = account_size * risk_per_trade
        
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            shares = max_loss / price_risk
            position_value = shares * entry_price
            position_size = position_value / account_size
        else:
            position_size = 0.02  # Default 2% position
        
        return min(position_size, 0.1)  # Cap at 10% of account
    
    def _generate_reasoning(self, signals: Dict[str, Any], decision_score: float, 
                          action: str) -> Dict[str, List[str]]:
        """Generate human-readable reasoning for the decision"""
        
        main_reasons = []
        supporting = []
        against = []
        
        # Main decision reasoning
        if action in ['BUY', 'WEAK_BUY']:
            main_reasons.append(f"Bullish signals dominate with {abs(decision_score):.1%} confidence")
        elif action in ['SELL', 'WEAK_SELL']:
            main_reasons.append(f"Bearish signals dominate with {abs(decision_score):.1%} confidence")
        else:
            main_reasons.append("Mixed signals suggest waiting for clearer direction")
        
        # Pattern reasoning
        pattern_signals = signals['patterns']
        if pattern_signals['direction'] != 'neutral':
            if pattern_signals['direction'] == 'bullish':
                supporting.extend([f"Bullish pattern: {p}" for p in pattern_signals['recent_patterns']])
            else:
                against.extend([f"Bearish pattern: {p}" for p in pattern_signals['recent_patterns']])
        
        # Technical indicator reasoning
        tech_signals = signals['technical']['signals']
        for signal in tech_signals:
            if 'Bullish' in signal or 'Oversold' in signal:
                supporting.append(signal)
            elif 'Bearish' in signal or 'Overbought' in signal:
                against.append(signal)
        
        # Momentum reasoning
        momentum_signals = signals['momentum']['signals']
        for signal in momentum_signals:
            if 'Bullish' in signal:
                supporting.append(signal)
            elif 'Bearish' in signal:
                against.append(signal)
        
        # Volume reasoning
        volume_signals = signals['volume']['signals']
        supporting.extend(volume_signals)
        
        return {
            'main_reasons': main_reasons,
            'supporting': supporting[:5],  # Limit to top 5
            'against': against[:5]  # Limit to top 5
        }
    
    def _determine_urgency(self, signals: Dict[str, Any], decision_score: float) -> str:
        """Determine urgency level of the trading decision"""
        
        # Check for breakout conditions
        volume_ratio = signals['volume'].get('volume_ratio', 1)
        momentum_score = abs(signals['momentum']['score'])
        
        if abs(decision_score) > 0.8 and volume_ratio > 2.0:
            return 'CRITICAL'
        elif abs(decision_score) > 0.6 and (volume_ratio > 1.5 or momentum_score > 0.5):
            return 'HIGH'
        elif abs(decision_score) > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_timeframe(self, signals: Dict[str, Any]) -> str:
        """Determine recommended holding timeframe"""
        
        momentum_score = abs(signals['momentum']['score'])
        pattern_count = signals['patterns']['pattern_count']
        
        if momentum_score > 0.7 or pattern_count >= 3:
            return 'Short-term (1-5 days)'
        elif momentum_score > 0.4:
            return 'Medium-term (1-2 weeks)'
        else:
            return 'Long-term (2+ weeks)'
    
    def _calculate_expected_return(self, levels: Dict[str, float], 
                                 current_price: float, action: str) -> float:
        """Calculate expected return percentage"""
        
        if action in ['BUY', 'WEAK_BUY']:
            return (levels['take_profit'] - current_price) / current_price
        elif action in ['SELL', 'WEAK_SELL']:
            return (current_price - levels['take_profit']) / current_price
        else:
            return 0.0
    
    def _calculate_max_risk(self, levels: Dict[str, float], current_price: float, 
                          action: str, position_size: float, account_size: float) -> float:
        """Calculate maximum risk percentage"""
        
        price_risk = abs(current_price - levels['stop_loss']) / current_price
        return price_risk * position_size
    
    def _determine_overall_trend(self, data: pd.DataFrame) -> str:
        """Determine overall market trend"""
        
        if len(data) < 50:
            return 'insufficient_data'
        
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return 'uptrend'
        elif current_price < sma_20 < sma_50:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _get_bb_position(self, price: float, bb: Dict[str, pd.Series]) -> str:
        """Get Bollinger Band position"""
        
        upper = bb['upper'].iloc[-1]
        lower = bb['lower'].iloc[-1]
        middle = bb['middle'].iloc[-1]
        
        if price >= upper:
            return 'above_upper'
        elif price <= lower:
            return 'below_lower'
        elif price > middle:
            return 'above_middle'
        else:
            return 'below_middle'