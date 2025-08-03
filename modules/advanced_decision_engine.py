import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

class AdvancedDecisionEngine:
    """Advanced decision engine with multi-factor analysis and confidence scoring"""
    
    def __init__(self):
        self.weight_factors = {
            'technical_indicators': 0.25,
            'trading_strategies': 0.25,
            'market_sentiment': 0.15,
            'volume_analysis': 0.15,
            'risk_metrics': 0.10,
            'fundamental_signals': 0.10
        }
        
        self.confidence_thresholds = {
            'very_high': 0.85,
            'high': 0.70,
            'medium': 0.55,
            'low': 0.40
        }
    
    def generate_comprehensive_decision(self, data: pd.DataFrame, symbol: str, 
                                      technical_signals: Dict, strategy_signals: Dict, 
                                      market_sentiment: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading decision with detailed analysis"""
        try:
            # Initialize decision framework
            decision = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'overall_signal': 'NEUTRAL',
                'confidence_score': 0.0,
                'confidence_level': 'LOW',
                'risk_level': 'MEDIUM',
                'position_sizing': 0.0,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'time_horizon': 'SHORT',
                'analysis_breakdown': {},
                'supporting_factors': [],
                'risk_factors': [],
                'market_conditions': {},
                'external_references': self._get_external_references(symbol)
            }
            
            current_price = data['Close'].iloc[-1]
            decision['entry_price'] = current_price
            
            # 1. Technical Indicators Analysis
            tech_score = self._analyze_technical_indicators(data, technical_signals)
            decision['analysis_breakdown']['technical_score'] = tech_score
            
            # 2. Trading Strategies Analysis
            strategy_score = self._analyze_trading_strategies(strategy_signals)
            decision['analysis_breakdown']['strategy_score'] = strategy_score
            
            # 3. Market Sentiment Analysis
            sentiment_score = self._analyze_market_sentiment(market_sentiment)
            decision['analysis_breakdown']['sentiment_score'] = sentiment_score
            
            # 4. Volume Analysis
            volume_score = self._analyze_volume_patterns(data)
            decision['analysis_breakdown']['volume_score'] = volume_score
            
            # 5. Risk Metrics Analysis
            risk_score = self._analyze_risk_metrics(data)
            decision['analysis_breakdown']['risk_score'] = risk_score
            
            # 6. Fundamental Signals (basic)
            fundamental_score = self._analyze_fundamental_signals(symbol, data)
            decision['analysis_breakdown']['fundamental_score'] = fundamental_score
            
            # Calculate weighted overall score
            overall_score = (
                tech_score['score'] * self.weight_factors['technical_indicators'] +
                strategy_score['score'] * self.weight_factors['trading_strategies'] +
                sentiment_score['score'] * self.weight_factors['market_sentiment'] +
                volume_score['score'] * self.weight_factors['volume_analysis'] +
                risk_score['score'] * self.weight_factors['risk_metrics'] +
                fundamental_score['score'] * self.weight_factors['fundamental_signals']
            )
            
            # Determine overall signal and confidence
            if overall_score > 0.6:
                decision['overall_signal'] = 'BUY'
            elif overall_score < -0.6:
                decision['overall_signal'] = 'SELL'
            else:
                decision['overall_signal'] = 'NEUTRAL'
            
            decision['confidence_score'] = abs(overall_score)
            decision['confidence_level'] = self._get_confidence_level(decision['confidence_score'])
            
            # Risk assessment
            decision['risk_level'] = self._assess_risk_level(data, decision['confidence_score'])
            
            # Position sizing recommendation
            decision['position_sizing'] = self._calculate_position_sizing(
                decision['confidence_score'], decision['risk_level']
            )
            
            # Entry and exit levels
            atr = self._calculate_atr(data)
            if decision['overall_signal'] == 'BUY':
                decision['stop_loss'] = current_price - (2 * atr)
                decision['take_profit'] = current_price + (3 * atr)
            elif decision['overall_signal'] == 'SELL':
                decision['stop_loss'] = current_price + (2 * atr)
                decision['take_profit'] = current_price - (3 * atr)
            
            # Supporting and risk factors
            decision['supporting_factors'] = self._identify_supporting_factors(
                tech_score, strategy_score, sentiment_score, volume_score, risk_score, fundamental_score
            )
            decision['risk_factors'] = self._identify_risk_factors(
                tech_score, strategy_score, sentiment_score, volume_score, risk_score, fundamental_score
            )
            
            # Market conditions
            decision['market_conditions'] = self._assess_market_conditions(data)
            
            # Time horizon recommendation
            decision['time_horizon'] = self._recommend_time_horizon(decision['confidence_score'], strategy_score)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error generating comprehensive decision: {str(e)}")
            return {'symbol': symbol, 'overall_signal': 'NEUTRAL', 'confidence_score': 0.0}
    
    def _analyze_technical_indicators(self, data: pd.DataFrame, signals: Dict) -> Dict[str, Any]:
        """Analyze technical indicators and return score"""
        try:
            score = 0.0
            positive_signals = 0
            negative_signals = 0
            total_signals = 0
            
            # Handle different signal formats
            if isinstance(signals, dict) and signals.get('signals'):
                signal_list = signals['signals']
                if isinstance(signal_list, list):
                    for signal in signal_list:
                        if isinstance(signal, dict):
                            total_signals += 1
                            signal_value = signal.get('signal', '')
                            if 'Buy' in str(signal_value) or signal_value == 'BUY':
                                positive_signals += 1
                                score += 0.2
                            elif 'Sell' in str(signal_value) or signal_value == 'SELL':
                                negative_signals += 1
                                score -= 0.2
            
            # Normalize score
            if total_signals > 0:
                score = score / total_signals * 5  # Scale to -1 to 1
            
            return {
                'score': max(-1, min(1, score)),
                'positive_signals': positive_signals,
                'negative_signals': negative_signals,
                'total_signals': total_signals,
                'strength': 'Strong' if abs(score) > 0.7 else 'Medium' if abs(score) > 0.4 else 'Weak'
            }
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {str(e)}")
            return {'score': 0.0, 'strength': 'Weak', 'positive_signals': 0, 'negative_signals': 0, 'total_signals': 0}
    
    def _analyze_trading_strategies(self, strategy_signals: Dict) -> Dict[str, Any]:
        """Analyze trading strategy signals"""
        try:
            if not strategy_signals.get('strategy_results'):
                return {'score': 0.0, 'strength': 'Weak'}
            
            buy_signals = len([s for s in strategy_signals['strategy_results'] if s.get('signal') == 'BUY'])
            sell_signals = len([s for s in strategy_signals['strategy_results'] if s.get('signal') == 'SELL'])
            total_signals = len(strategy_signals['strategy_results'])
            
            if total_signals == 0:
                return {'score': 0.0, 'strength': 'Weak'}
            
            # Calculate weighted score based on confidence
            weighted_score = 0.0
            for strategy in strategy_signals['strategy_results']:
                confidence = strategy.get('confidence', 0)
                if strategy.get('signal') == 'BUY':
                    weighted_score += confidence
                elif strategy.get('signal') == 'SELL':
                    weighted_score -= confidence
            
            # Normalize
            score = weighted_score / total_signals if total_signals > 0 else 0.0
            
            return {
                'score': max(-1, min(1, score)),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': total_signals,
                'consensus_confidence': strategy_signals.get('consensus_confidence', 0),
                'strength': 'Strong' if abs(score) > 0.7 else 'Medium' if abs(score) > 0.4 else 'Weak'
            }
        except Exception as e:
            logger.error(f"Error analyzing trading strategies: {str(e)}")
            return {'score': 0.0, 'strength': 'Weak', 'buy_signals': 0, 'sell_signals': 0, 'total_signals': 0, 'consensus_confidence': 0}
    
    def _analyze_market_sentiment(self, sentiment: Dict) -> Dict[str, Any]:
        """Analyze market sentiment"""
        try:
            # Extract sentiment score from the sentiment analysis
            bullish_signals = sentiment.get('bullish_signals', 0)
            bearish_signals = sentiment.get('bearish_signals', 0)
            total_signals = bullish_signals + bearish_signals
            
            if total_signals == 0:
                return {'score': 0.0, 'strength': 'Weak'}
            
            score = (bullish_signals - bearish_signals) / total_signals
            
            return {
                'score': max(-1, min(1, score)),
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'sentiment_direction': 'Bullish' if score > 0.2 else 'Bearish' if score < -0.2 else 'Neutral',
                'strength': 'Strong' if abs(score) > 0.6 else 'Medium' if abs(score) > 0.3 else 'Weak'
            }
        except Exception:
            return {'score': 0.0, 'strength': 'Weak'}
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            current_volume = data['Volume'].iloc[-1]
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Price-volume relationship
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
            
            score = 0.0
            if volume_ratio > 1.5:  # High volume
                if price_change > 0.01:  # Positive price movement
                    score = 0.8
                elif price_change < -0.01:  # Negative price movement
                    score = -0.8
                else:
                    score = 0.2  # High volume but no clear direction
            elif volume_ratio < 0.7:  # Low volume
                score = -0.3  # Generally negative for low volume
            
            return {
                'score': max(-1, min(1, score)),
                'volume_ratio': volume_ratio,
                'price_change': price_change * 100,
                'volume_trend': 'High' if volume_ratio > 1.5 else 'Low' if volume_ratio < 0.7 else 'Normal',
                'strength': 'Strong' if abs(score) > 0.7 else 'Medium' if abs(score) > 0.4 else 'Weak'
            }
        except Exception:
            return {'score': 0.0, 'strength': 'Weak'}
    
    def _analyze_risk_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk metrics"""
        try:
            # Volatility analysis
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # ATR analysis
            atr = self._calculate_atr(data)
            atr_pct = atr / data['Close'].iloc[-1] * 100
            
            # Risk score (lower volatility = higher score for risk management)
            volatility_score = max(-1, min(1, (0.3 - volatility) / 0.3))  # Normalize around 30% volatility
            atr_score = max(-1, min(1, (3 - atr_pct) / 3))  # Normalize around 3% ATR
            
            combined_score = (volatility_score + atr_score) / 2
            
            return {
                'score': combined_score,
                'volatility': volatility * 100,
                'atr_percentage': atr_pct,
                'risk_level': 'Low' if combined_score > 0.3 else 'High' if combined_score < -0.3 else 'Medium',
                'strength': 'Strong' if abs(combined_score) > 0.6 else 'Medium' if abs(combined_score) > 0.3 else 'Weak'
            }
        except Exception:
            return {'score': 0.0, 'strength': 'Weak'}
    
    def _analyze_fundamental_signals(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic fundamental analysis"""
        try:
            # Price trend analysis (proxy for fundamental strength)
            returns_30d = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30] if len(data) > 30 else 0
            returns_90d = (data['Close'].iloc[-1] - data['Close'].iloc[-90]) / data['Close'].iloc[-90] if len(data) > 90 else 0
            
            # Trend consistency
            trend_score = 0.0
            if returns_30d > 0.05 and returns_90d > 0.1:  # Strong uptrend
                trend_score = 0.8
            elif returns_30d < -0.05 and returns_90d < -0.1:  # Strong downtrend
                trend_score = -0.8
            elif returns_30d > 0 and returns_90d > 0:  # Positive trend
                trend_score = 0.4
            elif returns_30d < 0 and returns_90d < 0:  # Negative trend
                trend_score = -0.4
            
            return {
                'score': max(-1, min(1, trend_score)),
                'returns_30d': returns_30d * 100,
                'returns_90d': returns_90d * 100,
                'trend_direction': 'Bullish' if trend_score > 0.2 else 'Bearish' if trend_score < -0.2 else 'Neutral',
                'strength': 'Strong' if abs(trend_score) > 0.6 else 'Medium' if abs(trend_score) > 0.3 else 'Weak'
            }
        except Exception:
            return {'score': 0.0, 'strength': 'Weak'}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close_prev = abs(data['High'] - data['Close'].shift())
            low_close_prev = abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean().iloc[-1]
        except Exception:
            return 0.0
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score"""
        for level, threshold in self.confidence_thresholds.items():
            if score >= threshold:
                return level.replace('_', ' ').upper()
        return 'LOW'
    
    def _assess_risk_level(self, data: pd.DataFrame, confidence: float) -> str:
        """Assess risk level"""
        try:
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            
            if volatility > 0.4 or confidence < 0.4:
                return 'HIGH'
            elif volatility > 0.2 or confidence < 0.6:
                return 'MEDIUM'
            else:
                return 'LOW'
        except Exception:
            return 'MEDIUM'
    
    def _calculate_position_sizing(self, confidence: float, risk_level: str) -> float:
        """Calculate recommended position sizing as percentage of portfolio"""
        base_size = confidence * 0.1  # Base 10% max position
        
        risk_multiplier = {
            'LOW': 1.0,
            'MEDIUM': 0.7,
            'HIGH': 0.4
        }
        
        return min(0.1, base_size * risk_multiplier.get(risk_level, 0.7))
    
    def _identify_supporting_factors(self, tech: Dict, strategy: Dict, sentiment: Dict, 
                                   volume: Dict, risk: Dict, fundamental: Dict) -> List[str]:
        """Identify supporting factors for the decision"""
        factors = []
        
        if tech.get('strength') in ['Strong', 'Medium']:
            factors.append(f"Technical indicators show {tech['strength'].lower()} signals")
        
        if strategy.get('consensus_confidence', 0) > 0.6:
            factors.append(f"High strategy consensus ({strategy['consensus_confidence']:.1%})")
        
        if sentiment.get('strength') in ['Strong', 'Medium']:
            factors.append(f"Market sentiment is {sentiment.get('sentiment_direction', '').lower()}")
        
        if volume.get('volume_trend') == 'High':
            factors.append("High volume confirms price movement")
        
        if risk.get('risk_level') == 'Low':
            factors.append("Low volatility environment")
        
        if fundamental.get('strength') in ['Strong', 'Medium']:
            factors.append(f"Price trend shows {fundamental.get('trend_direction', '').lower()} momentum")
        
        return factors
    
    def _identify_risk_factors(self, tech: Dict, strategy: Dict, sentiment: Dict, 
                             volume: Dict, risk: Dict, fundamental: Dict) -> List[str]:
        """Identify risk factors"""
        factors = []
        
        if tech.get('strength') == 'Weak':
            factors.append("Weak technical signals")
        
        if strategy.get('consensus_confidence', 0) < 0.4:
            factors.append("Low strategy consensus")
        
        if sentiment.get('sentiment_direction') == 'Neutral':
            factors.append("Neutral market sentiment lacks conviction")
        
        if volume.get('volume_trend') == 'Low':
            factors.append("Low volume may indicate lack of interest")
        
        if risk.get('risk_level') == 'High':
            factors.append("High volatility increases risk")
        
        if fundamental.get('trend_direction') == 'Bearish':
            factors.append("Bearish price trend")
        
        return factors
    
    def _assess_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market conditions"""
        try:
            # Trend analysis
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else ma_20
            current_price = data['Close'].iloc[-1]
            
            if current_price > ma_20 > ma_50:
                trend = 'Strong Uptrend'
            elif current_price > ma_20:
                trend = 'Uptrend'
            elif current_price < ma_20 < ma_50:
                trend = 'Strong Downtrend'
            elif current_price < ma_20:
                trend = 'Downtrend'
            else:
                trend = 'Sideways'
            
            # Market phase
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
            if volatility > 3:
                phase = 'High Volatility'
            elif volatility < 1:
                phase = 'Low Volatility'
            else:
                phase = 'Normal Volatility'
            
            return {
                'trend': trend,
                'phase': phase,
                'volatility': volatility,
                'support_level': ma_20,
                'resistance_level': data['High'].rolling(20).max().iloc[-1]
            }
        except Exception:
            return {'trend': 'Unknown', 'phase': 'Unknown'}
    
    def _recommend_time_horizon(self, confidence: float, strategy_score: Dict) -> str:
        """Recommend time horizon based on analysis"""
        if confidence > 0.8:
            return 'MEDIUM'  # 1-4 weeks
        elif confidence > 0.6:
            return 'SHORT'   # 1-7 days
        else:
            return 'VERY_SHORT'  # Intraday
    
    def _get_external_references(self, symbol: str) -> Dict[str, str]:
        """Get external reference links for additional research"""
        return {
            'Yahoo Finance': f'https://finance.yahoo.com/quote/{symbol}',
            'Google Finance': f'https://www.google.com/finance/quote/{symbol}:NASDAQ',
            'MarketWatch': f'https://www.marketwatch.com/investing/stock/{symbol}',
            'Seeking Alpha': f'https://seekingalpha.com/symbol/{symbol}',
            'NASDAQ': f'https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}',
            'TradingView': f'https://www.tradingview.com/symbols/{symbol}',
            'Finviz': f'https://finviz.com/quote.ashx?t={symbol}',
            'SEC Filings': f'https://www.sec.gov/edgar/search/#/q={symbol}&entityName={symbol}',
            'StockTwits': f'https://stocktwits.com/symbol/{symbol}',
            'Benzinga': f'https://www.benzinga.com/quote/{symbol}',
            'StockChartsAI': f'https://stockchartsai.com/chart_search.php?symbol={symbol}'
        }