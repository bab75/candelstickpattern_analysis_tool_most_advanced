"""
Advanced Candlestick Formation Profit Analyzer
Professional-grade analysis for day trading, swing trading, and long-term profit confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class CandlestickProfitAnalyzer:
    """Analyzes candlestick formations for profit potential across different timeframes"""
    
    def __init__(self):
        # Profit confirmation patterns with historical success rates
        self.profit_patterns = {
            'day_trading': {
                'hammer': {'success_rate': 0.72, 'avg_profit': 1.8, 'time_to_profit': '2-4 hours'},
                'doji': {'success_rate': 0.58, 'avg_profit': 1.2, 'time_to_profit': '1-3 hours'},
                'engulfing': {'success_rate': 0.78, 'avg_profit': 2.4, 'time_to_profit': '1-2 hours'},
                'morning_star': {'success_rate': 0.82, 'avg_profit': 3.1, 'time_to_profit': '2-6 hours'},
                'evening_star': {'success_rate': 0.79, 'avg_profit': 2.8, 'time_to_profit': '2-6 hours'},
                'shooting_star': {'success_rate': 0.69, 'avg_profit': 2.1, 'time_to_profit': '1-4 hours'},
                'inverted_hammer': {'success_rate': 0.64, 'avg_profit': 1.9, 'time_to_profit': '2-5 hours'}
            },
            'swing_trading': {
                'hammer': {'success_rate': 0.76, 'avg_profit': 4.2, 'time_to_profit': '3-7 days'},
                'doji': {'success_rate': 0.62, 'avg_profit': 2.8, 'time_to_profit': '2-5 days'},
                'engulfing': {'success_rate': 0.84, 'avg_profit': 5.8, 'time_to_profit': '2-5 days'},
                'morning_star': {'success_rate': 0.87, 'avg_profit': 7.2, 'time_to_profit': '3-8 days'},
                'evening_star': {'success_rate': 0.85, 'avg_profit': 6.9, 'time_to_profit': '3-8 days'},
                'shooting_star': {'success_rate': 0.73, 'avg_profit': 4.8, 'time_to_profit': '2-6 days'},
                'three_white_soldiers': {'success_rate': 0.89, 'avg_profit': 8.4, 'time_to_profit': '5-10 days'},
                'three_black_crows': {'success_rate': 0.86, 'avg_profit': 7.9, 'time_to_profit': '5-10 days'}
            },
            'long_term': {
                'hammer': {'success_rate': 0.68, 'avg_profit': 12.5, 'time_to_profit': '2-8 weeks'},
                'doji': {'success_rate': 0.55, 'avg_profit': 8.2, 'time_to_profit': '3-12 weeks'},
                'engulfing': {'success_rate': 0.79, 'avg_profit': 18.3, 'time_to_profit': '3-10 weeks'},
                'morning_star': {'success_rate': 0.83, 'avg_profit': 22.7, 'time_to_profit': '4-12 weeks'},
                'evening_star': {'success_rate': 0.81, 'avg_profit': 20.4, 'time_to_profit': '4-12 weeks'},
                'three_white_soldiers': {'success_rate': 0.86, 'avg_profit': 25.8, 'time_to_profit': '6-16 weeks'},
                'three_black_crows': {'success_rate': 0.84, 'avg_profit': 23.1, 'time_to_profit': '6-16 weeks'}
            }
        }
        
        # Confirmation factors that increase success probability
        self.confirmation_factors = {
            'volume_surge': 1.15,  # 15% increase in success rate
            'trend_alignment': 1.20,  # 20% increase
            'support_resistance': 1.25,  # 25% increase
            'rsi_confirmation': 1.10,  # 10% increase
            'macd_alignment': 1.12,  # 12% increase
            'bollinger_position': 1.08  # 8% increase
        }
    
    def analyze_profit_potential(self, data: pd.DataFrame, patterns: List[Dict], 
                               trading_style: str = "swing_trading") -> Dict[str, Any]:
        """
        Comprehensive profit potential analysis based on candlestick formations
        
        Args:
            data: OHLCV DataFrame
            patterns: List of detected patterns
            trading_style: 'day_trading', 'swing_trading', or 'long_term'
        
        Returns:
            Detailed profit analysis with confidence scores
        """
        try:
            if not patterns or data.empty:
                return self._create_empty_analysis()
            
            current_price = data['Close'].iloc[-1]
            
            # Analyze each pattern for profit potential
            pattern_analyses = []
            for pattern in patterns:
                if isinstance(pattern, dict) and 'type' in pattern:
                    analysis = self._analyze_single_pattern(
                        pattern, data, trading_style, current_price
                    )
                    if analysis is not None:
                        pattern_analyses.append(analysis)
            
            # Generate overall profit assessment
            overall_analysis = self._generate_overall_assessment(
                pattern_analyses, data, trading_style, current_price
            )
            
            return {
                'trading_style': trading_style,
                'current_price': current_price,
                'pattern_analyses': pattern_analyses,
                'overall_assessment': overall_analysis,
                'risk_factors': self._identify_risk_factors(data, patterns),
                'confirmation_signals': self._check_confirmation_signals(data),
                'profit_targets': self._calculate_profit_targets(data, pattern_analyses, trading_style),
                'entry_exit_strategy': self._generate_entry_exit_strategy(pattern_analyses, trading_style)
            }
            
        except Exception as e:
            logger.error(f"Error in profit potential analysis: {e}")
            return self._create_empty_analysis()
    
    def _analyze_single_pattern(self, pattern: Dict, data: pd.DataFrame, 
                              trading_style: str, current_price: float) -> Dict[str, Any]:
        """Analyze profit potential for a single pattern"""
        try:
            pattern_type = pattern['type'].lower().replace(' ', '_')
            
            # Get base statistics for this pattern
            if trading_style not in self.profit_patterns:
                trading_style = 'swing_trading'
            
            style_patterns = self.profit_patterns[trading_style]
            if pattern_type not in style_patterns:
                # Use similar pattern or default
                pattern_type = self._find_similar_pattern(pattern_type, style_patterns)
            
            base_stats = style_patterns.get(pattern_type, style_patterns['hammer'])
            
            # Calculate confirmation factors
            confirmation_score = self._calculate_confirmation_score(data, pattern)
            
            # Adjust success rate based on confirmations
            adjusted_success_rate = min(0.95, base_stats['success_rate'] * confirmation_score)
            adjusted_profit = base_stats['avg_profit'] * confirmation_score
            
            # Calculate specific price targets
            entry_price = current_price
            if pattern.get('direction') == 'bullish':
                target_price = entry_price * (1 + adjusted_profit / 100)
                stop_loss = entry_price * 0.97  # 3% stop loss for bullish
            else:
                target_price = entry_price * (1 - adjusted_profit / 100)
                stop_loss = entry_price * 1.03  # 3% stop loss for bearish
            
            # Risk-reward ratio
            risk_amount = abs(entry_price - stop_loss)
            reward_amount = abs(target_price - entry_price)
            risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'pattern_type': pattern['type'],
                'direction': pattern.get('direction', 'neutral'),
                'success_rate': adjusted_success_rate,
                'expected_profit_pct': adjusted_profit,
                'time_to_profit': base_stats['time_to_profit'],
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'confidence_score': confirmation_score,
                'pattern_strength': pattern.get('strength', 'medium'),
                'volume_confirmation': pattern.get('volume_confirmed', False)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pattern {pattern}: {e}")
            return None
    
    def _calculate_confirmation_score(self, data: pd.DataFrame, pattern: Dict) -> float:
        """Calculate confirmation score based on technical indicators"""
        try:
            score = 1.0  # Base score
            
            # Volume confirmation
            if len(data) >= 20:
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:
                    score *= self.confirmation_factors['volume_surge']
            
            # RSI confirmation
            if len(data) >= 14:
                try:
                    rsi = self._calculate_rsi(data['Close'])
                    if isinstance(rsi, pd.Series) and not rsi.empty:
                        current_rsi = rsi.iloc[-1]
                        if pattern.get('direction') == 'bullish' and current_rsi < 40:
                            score *= self.confirmation_factors['rsi_confirmation']
                        elif pattern.get('direction') == 'bearish' and current_rsi > 60:
                            score *= self.confirmation_factors['rsi_confirmation']
                except:
                    pass
            
            # Trend alignment
            if len(data) >= 50:
                try:
                    ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                    ma_50 = data['Close'].rolling(50).mean().iloc[-1]
                    current_price = data['Close'].iloc[-1]
                
                    if pattern.get('direction') == 'bullish' and current_price > ma_20 > ma_50:
                        score *= self.confirmation_factors['trend_alignment']
                    elif pattern.get('direction') == 'bearish' and current_price < ma_20 < ma_50:
                        score *= self.confirmation_factors['trend_alignment']
                except:
                    pass
            
            # Support/Resistance levels
            try:
                high_20 = data['High'].rolling(20).max().iloc[-1]
                low_20 = data['Low'].rolling(20).min().iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                if pattern.get('direction') == 'bullish' and current_price <= low_20 * 1.02:
                    score *= self.confirmation_factors['support_resistance']
                elif pattern.get('direction') == 'bearish' and current_price >= high_20 * 0.98:
                    score *= self.confirmation_factors['support_resistance']
            except:
                pass
            
            return min(2.0, score)  # Cap at 2x
            
        except Exception as e:
            logger.error(f"Error calculating confirmation score: {e}")
            return 1.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            if not isinstance(prices, pd.Series):
                return pd.Series()
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series()
    
    def _generate_overall_assessment(self, pattern_analyses: List[Dict], 
                                   data: pd.DataFrame, trading_style: str, 
                                   current_price: float) -> Dict[str, Any]:
        """Generate overall profit assessment"""
        try:
            if not pattern_analyses:
                return {
                    'recommendation': 'WAIT',
                    'confidence': 0.0,
                    'expected_profit': 0.0,
                    'grade': 'F'
                }
            
            # Calculate weighted averages
            total_weight = sum(p['confidence_score'] for p in pattern_analyses)
            if total_weight == 0:
                return {'recommendation': 'WAIT', 'confidence': 0.0, 'expected_profit': 0.0, 'grade': 'F'}
            
            weighted_success = sum(p['success_rate'] * p['confidence_score'] for p in pattern_analyses) / total_weight
            weighted_profit = sum(p['expected_profit_pct'] * p['confidence_score'] for p in pattern_analyses) / total_weight
            weighted_rr = sum(p['risk_reward_ratio'] * p['confidence_score'] for p in pattern_analyses) / total_weight
            
            # Determine recommendation
            if weighted_success >= 0.75 and weighted_rr >= 2.0:
                recommendation = 'STRONG BUY' if any(p['direction'] == 'bullish' for p in pattern_analyses) else 'STRONG SELL'
                grade = 'A+'
            elif weighted_success >= 0.65 and weighted_rr >= 1.5:
                recommendation = 'BUY' if any(p['direction'] == 'bullish' for p in pattern_analyses) else 'SELL'
                grade = 'A' if weighted_success >= 0.70 else 'B+'
            elif weighted_success >= 0.55:
                recommendation = 'WEAK BUY' if any(p['direction'] == 'bullish' for p in pattern_analyses) else 'WEAK SELL'
                grade = 'B' if weighted_success >= 0.60 else 'C+'
            else:
                recommendation = 'WAIT'
                grade = 'C' if weighted_success >= 0.50 else 'D'
            
            return {
                'recommendation': recommendation,
                'confidence': weighted_success,
                'expected_profit_pct': weighted_profit,
                'risk_reward_ratio': weighted_rr,
                'grade': grade,
                'pattern_count': len(pattern_analyses),
                'bullish_patterns': sum(1 for p in pattern_analyses if p['direction'] == 'bullish'),
                'bearish_patterns': sum(1 for p in pattern_analyses if p['direction'] == 'bearish')
            }
            
        except Exception as e:
            logger.error(f"Error generating overall assessment: {e}")
            return {'recommendation': 'WAIT', 'confidence': 0.0, 'expected_profit': 0.0, 'grade': 'F'}
    
    def _identify_risk_factors(self, data: pd.DataFrame, patterns: List[Dict]) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        try:
            # Volatility risk
            if len(data) >= 20:
                volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                if volatility > 5:
                    risks.append(f"High volatility: {volatility:.1f}% (20-day)")
            
            # Volume risk
            if len(data) >= 20:
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                if current_volume < avg_volume * 0.5:
                    risks.append("Low volume confirmation")
            
            # Conflicting patterns
            bullish_count = sum(1 for p in patterns if isinstance(p, dict) and p.get('direction') == 'bullish')
            bearish_count = sum(1 for p in patterns if isinstance(p, dict) and p.get('direction') == 'bearish')
            
            if bullish_count > 0 and bearish_count > 0:
                risks.append("Conflicting pattern signals")
            
            # Near resistance/support
            if len(data) >= 20:
                high_20 = data['High'].rolling(20).max().iloc[-1]
                low_20 = data['Low'].rolling(20).min().iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                if current_price >= high_20 * 0.98:
                    risks.append("Near 20-day resistance level")
                elif current_price <= low_20 * 1.02:
                    risks.append("Near 20-day support level")
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
        
        return risks
    
    def _check_confirmation_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for additional confirmation signals"""
        confirmations = {
            'volume_trend': 'neutral',
            'price_momentum': 'neutral',
            'technical_indicators': []
        }
        
        try:
            # Volume trend
            if len(data) >= 10:
                recent_volume = data['Volume'].tail(5).mean()
                previous_volume = data['Volume'].tail(10).head(5).mean()
                if recent_volume > previous_volume * 1.2:
                    confirmations['volume_trend'] = 'increasing'
                elif recent_volume < previous_volume * 0.8:
                    confirmations['volume_trend'] = 'decreasing'
            
            # Price momentum
            if len(data) >= 5:
                recent_returns = data['Close'].pct_change().tail(5).mean()
                if recent_returns > 0.01:
                    confirmations['price_momentum'] = 'bullish'
                elif recent_returns < -0.01:
                    confirmations['price_momentum'] = 'bearish'
            
            # Technical confirmations
            if len(data) >= 20:
                # Moving average alignment
                ma_5 = data['Close'].rolling(5).mean().iloc[-1]
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                if current_price > ma_5 > ma_20:
                    confirmations['technical_indicators'].append("Bullish MA alignment")
                elif current_price < ma_5 < ma_20:
                    confirmations['technical_indicators'].append("Bearish MA alignment")
            
        except Exception as e:
            logger.error(f"Error checking confirmation signals: {e}")
        
        return confirmations
    
    def _calculate_profit_targets(self, data: pd.DataFrame, pattern_analyses: List[Dict], 
                                trading_style: str) -> Dict[str, float]:
        """Calculate specific profit targets"""
        targets = {}
        
        try:
            current_price = data['Close'].iloc[-1]
            
            if pattern_analyses:
                # Conservative target (lowest expected profit)
                conservative = min(p['expected_profit_pct'] for p in pattern_analyses)
                targets['conservative'] = current_price * (1 + conservative / 100)
                
                # Optimistic target (highest expected profit)
                optimistic = max(p['expected_profit_pct'] for p in pattern_analyses)
                targets['optimistic'] = current_price * (1 + optimistic / 100)
                
                # Realistic target (weighted average)
                total_weight = sum(p['confidence_score'] for p in pattern_analyses)
                realistic = sum(p['expected_profit_pct'] * p['confidence_score'] for p in pattern_analyses) / total_weight
                targets['realistic'] = current_price * (1 + realistic / 100)
            
        except Exception as e:
            logger.error(f"Error calculating profit targets: {e}")
        
        return targets
    
    def _generate_entry_exit_strategy(self, pattern_analyses: List[Dict], 
                                    trading_style: str) -> Dict[str, Any]:
        """Generate entry and exit strategy recommendations"""
        strategy = {
            'entry_timing': 'immediate',
            'position_sizing': 'moderate',
            'exit_strategy': 'trailing_stop',
            'timeframe': 'medium_term'
        }
        
        try:
            if not pattern_analyses:
                return strategy
            
            # Best pattern analysis
            best_pattern = max(pattern_analyses, key=lambda x: x['confidence_score'])
            
            # Entry timing based on pattern strength
            if best_pattern['confidence_score'] > 1.5:
                strategy['entry_timing'] = 'immediate'
            elif best_pattern['confidence_score'] > 1.2:
                strategy['entry_timing'] = 'on_breakout'
            else:
                strategy['entry_timing'] = 'wait_for_confirmation'
            
            # Position sizing based on success rate
            avg_success = sum(p['success_rate'] for p in pattern_analyses) / len(pattern_analyses)
            if avg_success > 0.8:
                strategy['position_sizing'] = 'aggressive'
            elif avg_success > 0.65:
                strategy['position_sizing'] = 'moderate'
            else:
                strategy['position_sizing'] = 'conservative'
            
            # Exit strategy based on risk-reward
            avg_rr = sum(p['risk_reward_ratio'] for p in pattern_analyses) / len(pattern_analyses)
            if avg_rr > 3.0:
                strategy['exit_strategy'] = 'profit_ladder'
            elif avg_rr > 2.0:
                strategy['exit_strategy'] = 'trailing_stop'
            else:
                strategy['exit_strategy'] = 'fixed_target'
            
            # Timeframe alignment
            if trading_style == 'day_trading':
                strategy['timeframe'] = 'intraday'
            elif trading_style == 'swing_trading':
                strategy['timeframe'] = 'short_term'
            else:
                strategy['timeframe'] = 'long_term'
            
        except Exception as e:
            logger.error(f"Error generating entry/exit strategy: {e}")
        
        return strategy
    
    def _find_similar_pattern(self, pattern_type: str, available_patterns: Dict) -> str:
        """Find similar pattern if exact match not found"""
        pattern_map = {
            'hanging_man': 'hammer',
            'inverted_hammer': 'shooting_star',
            'bullish_engulfing': 'engulfing',
            'bearish_engulfing': 'engulfing',
            'bullish_harami': 'engulfing',
            'bearish_harami': 'engulfing'
        }
        
        return pattern_map.get(pattern_type, 'hammer')
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis structure"""
        return {
            'trading_style': 'swing_trading',
            'current_price': 0.0,
            'pattern_analyses': [],
            'overall_assessment': {
                'recommendation': 'WAIT',
                'confidence': 0.0,
                'expected_profit_pct': 0.0,
                'grade': 'N/A'
            },
            'risk_factors': ['No patterns detected'],
            'confirmation_signals': {},
            'profit_targets': {},
            'entry_exit_strategy': {}
        }