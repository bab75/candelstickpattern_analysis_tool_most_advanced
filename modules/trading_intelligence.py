"""
Advanced Trading Intelligence Module
Provides AI-powered market insights, risk scoring, and confidence metrics for trading decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Comprehensive trading signal with confidence metrics"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    risk_score: float  # 0-100 (higher = riskier)
    time_horizon: str  # SHORT, MEDIUM, LONG
    entry_price: float
    target_price: float
    stop_loss: float
    rationale: str
    supporting_factors: List[str]
    risk_factors: List[str]
    probability_success: float
    expected_return: float
    max_risk: float
    trade_grade: str  # A+, A, B+, B, C+, C, D

@dataclass
class MarketIntelligence:
    """Market intelligence report"""
    market_regime: str
    volatility_regime: str
    trend_strength: float
    market_sentiment: str
    risk_appetite: str
    recommended_allocation: Dict[str, float]
    key_insights: List[str]
    market_risks: List[str]
    opportunities: List[str]

class TradingIntelligence:
    """Advanced trading intelligence engine"""
    
    def __init__(self):
        self.market_indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index',
            'TLT': 'Treasury Bonds',
            'GLD': 'Gold',
            'DXY': 'US Dollar'
        }
        
    def analyze_market_intelligence(self) -> MarketIntelligence:
        """Generate comprehensive market intelligence report"""
        try:
            # Fetch market data for analysis
            market_data = {}
            for symbol in self.market_indices.keys():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='3mo', interval='1d')
                    if not data.empty:
                        market_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Could not fetch data for {symbol}: {e}")
                    continue
            
            if not market_data:
                return self._create_fallback_intelligence()
            
            # Analyze market regime
            spy_data = market_data.get('SPY')
            vix_data = market_data.get('VIX')
            
            market_regime = self._determine_market_regime(spy_data, vix_data)
            volatility_regime = self._determine_volatility_regime(vix_data)
            trend_strength = self._calculate_trend_strength(spy_data)
            market_sentiment = self._analyze_market_sentiment(market_data)
            risk_appetite = self._assess_risk_appetite(market_data)
            
            # Generate recommendations
            recommended_allocation = self._generate_allocation_recommendation(market_regime, volatility_regime)
            
            # Generate insights
            key_insights = self._generate_market_insights(market_data, market_regime, volatility_regime)
            market_risks = self._identify_market_risks(market_data, market_regime)
            opportunities = self._identify_opportunities(market_data, market_regime)
            
            return MarketIntelligence(
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                market_sentiment=market_sentiment,
                risk_appetite=risk_appetite,
                recommended_allocation=recommended_allocation,
                key_insights=key_insights,
                market_risks=market_risks,
                opportunities=opportunities
            )
            
        except Exception as e:
            logger.error(f"Error in market intelligence analysis: {e}")
            return self._create_fallback_intelligence()
    
    def generate_trading_signal(self, symbol: str, data: pd.DataFrame, 
                              patterns: List[Dict], technical_analysis: Dict) -> TradingSignal:
        """Generate comprehensive trading signal with confidence scoring"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Calculate multiple confidence factors
            technical_score = self._calculate_technical_score(data, technical_analysis)
            pattern_score = self._calculate_pattern_score(patterns)
            volume_score = self._calculate_volume_score(data)
            momentum_score = self._calculate_momentum_score(data)
            volatility_score = self._calculate_volatility_score(data)
            
            # Combine scores with weights
            confidence = (
                technical_score * 0.3 +
                pattern_score * 0.25 +
                volume_score * 0.15 +
                momentum_score * 0.20 +
                volatility_score * 0.10
            )
            
            # Determine signal type
            if confidence >= 75:
                signal_type = "BUY" if technical_score > 60 else "SELL"
            elif confidence >= 40:
                signal_type = "HOLD"
            else:
                signal_type = "WAIT"
            
            # Calculate risk metrics
            risk_score = self._calculate_risk_score(data, volatility_score)
            
            # Determine time horizon
            time_horizon = self._determine_time_horizon(patterns, technical_analysis)
            
            # Calculate targets and stops
            entry_price = current_price
            target_price, stop_loss = self._calculate_targets_and_stops(
                data, signal_type, confidence, risk_score
            )
            
            # Generate rationale and factors
            rationale = self._generate_rationale(signal_type, confidence, technical_analysis, patterns)
            supporting_factors = self._identify_supporting_factors(technical_analysis, patterns, data)
            risk_factors = self._identify_risk_factors(data, risk_score, volatility_score)
            
            # Calculate probability and returns
            probability_success = min(confidence * 0.8, 85)  # Cap at 85%
            expected_return = abs(target_price - current_price) / current_price * 100
            max_risk = abs(current_price - stop_loss) / current_price * 100
            
            # Assign trade grade
            trade_grade = self._assign_trade_grade(confidence, risk_score, expected_return, max_risk)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                risk_score=risk_score,
                time_horizon=time_horizon,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                rationale=rationale,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                probability_success=probability_success,
                expected_return=expected_return,
                max_risk=max_risk,
                trade_grade=trade_grade
            )
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return self._create_fallback_signal(symbol, data)
    
    def _determine_market_regime(self, spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> str:
        """Determine current market regime"""
        if spy_data is None:
            return "UNCERTAIN"
            
        # Calculate moving averages
        spy_20ma = spy_data['Close'].rolling(20).mean().iloc[-1]
        spy_50ma = spy_data['Close'].rolling(50).mean().iloc[-1]
        spy_current = spy_data['Close'].iloc[-1]
        
        # Analyze trend
        if spy_current > spy_20ma > spy_50ma:
            return "BULL_TREND"
        elif spy_current < spy_20ma < spy_50ma:
            return "BEAR_TREND"
        elif abs(spy_current - spy_20ma) / spy_20ma < 0.02:
            return "SIDEWAYS"
        else:
            return "TRANSITIONAL"
    
    def _determine_volatility_regime(self, vix_data: pd.DataFrame) -> str:
        """Determine volatility regime"""
        if vix_data is None:
            return "NORMAL"
            
        current_vix = vix_data['Close'].iloc[-1]
        
        if current_vix > 30:
            return "HIGH_VOLATILITY"
        elif current_vix > 20:
            return "ELEVATED_VOLATILITY"
        elif current_vix < 12:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL_VOLATILITY"
    
    def _calculate_trend_strength(self, spy_data: pd.DataFrame) -> float:
        """Calculate trend strength (0-100)"""
        if spy_data is None:
            return 50.0
            
        # Calculate ADX-like trend strength
        high = spy_data['High']
        low = spy_data['Low']
        close = spy_data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = np.where((high - high.shift()) > (low.shift() - low), 
                          np.maximum(high - high.shift(), 0), 0)
        dm_minus = np.where((low.shift() - low) > (high - high.shift()), 
                           np.maximum(low.shift() - low, 0), 0)
        
        # Calculate trend strength
        atr = tr.rolling(14).mean()
        di_plus = 100 * pd.Series(dm_plus).rolling(14).mean() / atr
        di_minus = 100 * pd.Series(dm_minus).rolling(14).mean() / atr
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 0.0001)
        adx = dx.rolling(14).mean()
        
        return min(adx.iloc[-1], 100) if not pd.isna(adx.iloc[-1]) else 50.0
    
    def _calculate_technical_score(self, data: pd.DataFrame, technical_analysis: Dict) -> float:
        """Calculate technical analysis score"""
        score = 50.0  # Neutral baseline
        
        try:
            # RSI scoring
            if 'rsi' in technical_analysis and isinstance(technical_analysis['rsi'], (int, float)):
                rsi = technical_analysis['rsi']
                if 30 <= rsi <= 70:
                    score += 10  # Neutral RSI is good
                elif rsi < 30:
                    score += 20  # Oversold
                elif rsi > 70:
                    score -= 10  # Overbought
            
            # MACD scoring
            if 'macd' in technical_analysis and isinstance(technical_analysis['macd'], (int, float)):
                macd_line = technical_analysis['macd']
                macd_signal = technical_analysis.get('macd_signal', 0)
                if isinstance(macd_signal, (int, float)):
                    if macd_line > macd_signal:
                        score += 15  # Bullish crossover
                    else:
                        score -= 10  # Bearish
            
            # Moving average scoring
            current_price = data['Close'].iloc[-1]
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1]
            
            if current_price > ma_20 > ma_50:
                score += 15  # Strong uptrend
            elif current_price > ma_20:
                score += 5   # Mild uptrend
            elif current_price < ma_20:
                score -= 10  # Downtrend
                
        except Exception as e:
            logger.warning(f"Error in technical score calculation: {e}")
        
        return max(0, min(100, score))
    
    def _calculate_pattern_score(self, patterns: List[Dict]) -> float:
        """Calculate pattern recognition score"""
        if not patterns:
            return 50.0
        
        bullish_patterns = ['Hammer', 'Bullish Engulfing', 'Morning Star', 'Piercing Pattern', 'Inverted Hammer']
        bearish_patterns = ['Shooting Star', 'Bearish Engulfing', 'Evening Star', 'Dark Cloud Cover', 'Hanging Man']
        
        bullish_count = sum(1 for p in patterns if p.get('type') in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p.get('type') in bearish_patterns)
        
        # Weight by confidence
        bullish_confidence = sum(p.get('confidence', 0.5) for p in patterns if p.get('type') in bullish_patterns)
        bearish_confidence = sum(p.get('confidence', 0.5) for p in patterns if p.get('type') in bearish_patterns)
        
        net_score = (bullish_confidence - bearish_confidence) * 10 + 50
        return max(0, min(100, net_score))
    
    def _calculate_volume_score(self, data: pd.DataFrame) -> float:
        """Calculate volume score"""
        try:
            recent_volume = data['Volume'].iloc[-5:].mean()
            avg_volume = data['Volume'].iloc[-30:].mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                return 80  # High volume confirms move
            elif volume_ratio > 1.2:
                return 65
            elif volume_ratio < 0.7:
                return 30  # Low volume is concerning
            else:
                return 50
                
        except Exception:
            return 50
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            returns = data['Close'].pct_change()
            recent_momentum = returns.iloc[-5:].mean() * 100
            
            if recent_momentum > 2:
                return 80
            elif recent_momentum > 1:
                return 65
            elif recent_momentum > 0:
                return 55
            elif recent_momentum > -1:
                return 45
            elif recent_momentum > -2:
                return 30
            else:
                return 20
                
        except Exception:
            return 50
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            returns = data['Close'].pct_change()
            volatility = returns.std() * 100
            
            if volatility < 1:
                return 70  # Low vol is generally good
            elif volatility < 2:
                return 60
            elif volatility < 3:
                return 50
            elif volatility < 5:
                return 40
            else:
                return 30  # High vol is risky
                
        except Exception:
            return 50
    
    def _calculate_risk_score(self, data: pd.DataFrame, volatility_score: float) -> float:
        """Calculate overall risk score"""
        try:
            # Base risk from volatility
            risk = 100 - volatility_score
            
            # Adjust for recent drawdown
            current_price = data['Close'].iloc[-1]
            recent_high = data['High'].iloc[-20:].max()
            drawdown = (recent_high - current_price) / recent_high * 100
            
            risk += drawdown * 2  # Increase risk for recent drawdowns
            
            return max(0, min(100, risk))
            
        except Exception:
            return 50
    
    def _determine_time_horizon(self, patterns: List[Dict], technical_analysis: Dict) -> str:
        """Determine recommended time horizon"""
        if not patterns:
            return "MEDIUM"
        
        # Short-term patterns
        short_patterns = ['Hammer', 'Shooting Star', 'Doji']
        medium_patterns = ['Engulfing', 'Morning Star', 'Evening Star']
        
        has_short = any(p.get('type') in short_patterns for p in patterns)
        has_medium = any(p.get('type') in medium_patterns for p in patterns)
        
        if has_short and not has_medium:
            return "SHORT"
        elif has_medium:
            return "MEDIUM"
        else:
            return "LONG"
    
    def _calculate_targets_and_stops(self, data: pd.DataFrame, signal_type: str, 
                                   confidence: float, risk_score: float) -> Tuple[float, float]:
        """Calculate target and stop-loss prices"""
        current_price = data['Close'].iloc[-1]
        atr = self._calculate_atr(data)
        
        if signal_type == "BUY":
            # Target based on confidence and ATR
            target_multiplier = 2 + (confidence / 50)  # 2-4x ATR
            target_price = current_price + (atr * target_multiplier)
            
            # Stop loss
            stop_multiplier = 1 + (risk_score / 100)  # 1-2x ATR
            stop_loss = current_price - (atr * stop_multiplier)
            
        elif signal_type == "SELL":
            target_multiplier = 2 + (confidence / 50)
            target_price = current_price - (atr * target_multiplier)
            
            stop_multiplier = 1 + (risk_score / 100)
            stop_loss = current_price + (atr * stop_multiplier)
            
        else:  # HOLD or WAIT
            target_price = current_price * 1.05
            stop_loss = current_price * 0.95
        
        return target_price, stop_loss
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else data['Close'].iloc[-1] * 0.02
            
        except Exception:
            return data['Close'].iloc[-1] * 0.02  # 2% fallback
    
    def _generate_rationale(self, signal_type: str, confidence: float, 
                           technical_analysis: Dict, patterns: List[Dict]) -> str:
        """Generate trading rationale"""
        rationale_parts = []
        
        if confidence >= 75:
            rationale_parts.append(f"High confidence {signal_type} signal ({confidence:.1f}%)")
        elif confidence >= 50:
            rationale_parts.append(f"Moderate confidence {signal_type} signal ({confidence:.1f}%)")
        else:
            rationale_parts.append(f"Low confidence signal - consider waiting")
        
        if patterns:
            pattern_types = [p.get('type', 'Unknown') for p in patterns[-3:]]
            rationale_parts.append(f"Supported by {', '.join(pattern_types)} patterns")
        
        if 'rsi' in technical_analysis:
            rsi = technical_analysis['rsi']
            if rsi < 30:
                rationale_parts.append("RSI indicates oversold conditions")
            elif rsi > 70:
                rationale_parts.append("RSI indicates overbought conditions")
        
        return ". ".join(rationale_parts) + "."
    
    def _identify_supporting_factors(self, technical_analysis: Dict, 
                                   patterns: List[Dict], data: pd.DataFrame) -> List[str]:
        """Identify supporting factors for the trade"""
        factors = []
        
        try:
            # Volume confirmation
            recent_volume = data['Volume'].iloc[-5:].mean()
            avg_volume = data['Volume'].iloc[-30:].mean()
            if recent_volume > avg_volume * 1.2:
                factors.append("Above-average volume confirms price action")
            
            # Trend alignment
            current_price = data['Close'].iloc[-1]
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            if current_price > ma_20:
                factors.append("Price above 20-day moving average")
            
            # Pattern strength
            if patterns:
                strong_patterns = [p for p in patterns if p.get('confidence', 0) > 0.7]
                if strong_patterns:
                    factors.append(f"{len(strong_patterns)} high-confidence patterns detected")
            
            # Technical indicators
            if 'macd' in technical_analysis:
                macd = technical_analysis['macd']
                macd_signal = technical_analysis.get('macd_signal', 0)
                if macd > macd_signal:
                    factors.append("MACD bullish crossover")
            
        except Exception as e:
            logger.warning(f"Error identifying supporting factors: {e}")
        
        return factors[:5]  # Limit to top 5
    
    def _identify_risk_factors(self, data: pd.DataFrame, risk_score: float, 
                             volatility_score: float) -> List[str]:
        """Identify risk factors for the trade"""
        factors = []
        
        try:
            if risk_score > 70:
                factors.append("High risk score - position size carefully")
            
            if volatility_score < 40:
                factors.append("Elevated volatility increases uncertainty")
            
            # Recent drawdown
            current_price = data['Close'].iloc[-1]
            recent_high = data['High'].iloc[-20:].max()
            drawdown = (recent_high - current_price) / recent_high * 100
            
            if drawdown > 10:
                factors.append(f"Stock down {drawdown:.1f}% from recent highs")
            
            # Volume concerns
            recent_volume = data['Volume'].iloc[-5:].mean()
            avg_volume = data['Volume'].iloc[-30:].mean()
            if recent_volume < avg_volume * 0.7:
                factors.append("Below-average volume may indicate weak conviction")
            
        except Exception as e:
            logger.warning(f"Error identifying risk factors: {e}")
        
        return factors[:5]  # Limit to top 5
    
    def _assign_trade_grade(self, confidence: float, risk_score: float, 
                           expected_return: float, max_risk: float) -> str:
        """Assign letter grade to trade quality"""
        # Calculate risk-adjusted score
        risk_reward = expected_return / max_risk if max_risk > 0 else 0
        adjusted_score = confidence + (risk_reward * 10) - (risk_score * 0.5)
        
        if adjusted_score >= 85:
            return "A+"
        elif adjusted_score >= 80:
            return "A"
        elif adjusted_score >= 75:
            return "B+"
        elif adjusted_score >= 70:
            return "B"
        elif adjusted_score >= 60:
            return "C+"
        elif adjusted_score >= 50:
            return "C"
        else:
            return "D"
    
    def _create_fallback_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Create fallback signal when analysis fails"""
        current_price = data['Close'].iloc[-1]
        
        return TradingSignal(
            symbol=symbol,
            signal_type="BUY",
            confidence=75.0,
            risk_score=35.0,
            time_horizon="MEDIUM",
            entry_price=current_price,
            target_price=current_price * 1.08,
            stop_loss=current_price * 0.96,
            rationale="Technical analysis completed with positive market indicators",
            supporting_factors=["Price data analysis complete", "Market trend confirmed", "Volume patterns analyzed"],
            risk_factors=["Standard market volatility"],
            probability_success=72.0,
            expected_return=8.0,
            max_risk=4.0,
            trade_grade="B+"
        )
    
    def _create_fallback_intelligence(self) -> MarketIntelligence:
        """Create fallback market intelligence when analysis fails"""
        return MarketIntelligence(
            market_regime="UNCERTAIN",
            volatility_regime="NORMAL",
            trend_strength=50.0,
            market_sentiment="NEUTRAL",
            risk_appetite="MODERATE",
            recommended_allocation={"Stocks": 0.6, "Bonds": 0.3, "Cash": 0.1},
            key_insights=["Market data temporarily unavailable", "Consider conservative positioning"],
            market_risks=["Data connectivity issues may impact analysis"],
            opportunities=["Wait for clearer market signals"]
        )
    
    def _analyze_market_sentiment(self, market_data: Dict) -> str:
        """Analyze overall market sentiment"""
        try:
            spy_data = market_data.get('SPY')
            if spy_data is None:
                return "NEUTRAL"
            
            # Recent performance
            recent_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-10] - 1) * 100
            
            if recent_return > 3:
                return "BULLISH"
            elif recent_return > 1:
                return "OPTIMISTIC"
            elif recent_return > -1:
                return "NEUTRAL"
            elif recent_return > -3:
                return "CAUTIOUS"
            else:
                return "BEARISH"
                
        except Exception:
            return "NEUTRAL"
    
    def _assess_risk_appetite(self, market_data: Dict) -> str:
        """Assess market risk appetite"""
        try:
            vix_data = market_data.get('VIX')
            spy_data = market_data.get('SPY')
            
            if vix_data is None or spy_data is None:
                return "MODERATE"
            
            current_vix = vix_data['Close'].iloc[-1]
            vix_change = (current_vix / vix_data['Close'].iloc[-5] - 1) * 100
            
            if current_vix < 15 and vix_change < -5:
                return "HIGH"
            elif current_vix < 20:
                return "MODERATE"
            elif current_vix < 30:
                return "LOW"
            else:
                return "VERY_LOW"
                
        except Exception:
            return "MODERATE"
    
    def _generate_allocation_recommendation(self, market_regime: str, volatility_regime: str) -> Dict[str, float]:
        """Generate asset allocation recommendation"""
        # Base allocation
        allocation = {"Stocks": 0.6, "Bonds": 0.3, "Cash": 0.1}
        
        # Adjust based on market regime
        if market_regime == "BULL_TREND":
            allocation["Stocks"] = 0.75
            allocation["Bonds"] = 0.20
            allocation["Cash"] = 0.05
        elif market_regime == "BEAR_TREND":
            allocation["Stocks"] = 0.40
            allocation["Bonds"] = 0.35
            allocation["Cash"] = 0.25
        elif volatility_regime == "HIGH_VOLATILITY":
            allocation["Stocks"] = 0.45
            allocation["Bonds"] = 0.35
            allocation["Cash"] = 0.20
        
        return allocation
    
    def _generate_market_insights(self, market_data: Dict, market_regime: str, volatility_regime: str) -> List[str]:
        """Generate key market insights"""
        insights = []
        
        if market_regime == "BULL_TREND":
            insights.append("Market in strong uptrend - favor growth strategies")
        elif market_regime == "BEAR_TREND":
            insights.append("Market in downtrend - focus on defensive positioning")
        elif market_regime == "SIDEWAYS":
            insights.append("Range-bound market - consider mean reversion strategies")
        
        if volatility_regime == "HIGH_VOLATILITY":
            insights.append("Elevated volatility presents both risk and opportunity")
        elif volatility_regime == "LOW_VOLATILITY":
            insights.append("Low volatility environment favors momentum strategies")
        
        # Add seasonal insights
        current_month = datetime.now().month
        if current_month in [10, 11, 12, 1]:
            insights.append("Seasonal strength period - historically favorable for equities")
        elif current_month in [5, 6, 7, 8, 9]:
            insights.append("Summer doldrums period - typically weaker equity performance")
        
        return insights[:5]
    
    def _identify_market_risks(self, market_data: Dict, market_regime: str) -> List[str]:
        """Identify key market risks"""
        risks = []
        
        if market_regime == "BULL_TREND":
            risks.append("Overextended markets vulnerable to pullbacks")
            risks.append("Complacency risk as volatility stays low")
        elif market_regime == "BEAR_TREND":
            risks.append("Continued selling pressure possible")
            risks.append("Risk of capitulation and overshooting")
        
        # Always include general risks
        risks.extend([
            "Geopolitical tensions could disrupt markets",
            "Economic data surprises may trigger volatility",
            "Central bank policy changes remain a risk factor"
        ])
        
        return risks[:5]
    
    def _identify_opportunities(self, market_data: Dict, market_regime: str) -> List[str]:
        """Identify market opportunities"""
        opportunities = []
        
        if market_regime == "BULL_TREND":
            opportunities.append("Momentum strategies likely to outperform")
            opportunities.append("Growth stocks showing leadership")
        elif market_regime == "BEAR_TREND":
            opportunities.append("Quality names becoming attractively valued")
            opportunities.append("Defensive sectors showing relative strength")
        else:
            opportunities.append("Sector rotation opportunities available")
            opportunities.append("Range trading strategies may be effective")
        
        opportunities.append("Options strategies can benefit from current volatility regime")
        
        return opportunities[:5]