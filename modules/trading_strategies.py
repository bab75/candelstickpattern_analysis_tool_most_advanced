import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingStrategies:
    """Professional trading strategies and signal generation"""
    
    def __init__(self):
        self.strategies = {
            'Mean Reversion': self._mean_reversion_strategy,
            'Momentum Breakout': self._momentum_breakout_strategy,
            'Trend Following': self._trend_following_strategy,
            'Swing Trading': self._swing_trading_strategy,
            'Scalping': self._scalping_strategy,
            'Gap Trading': self._gap_trading_strategy,
            'Volume Spike': self._volume_spike_strategy,
            'Support Resistance': self._support_resistance_strategy
        }
    
    def _mean_reversion_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Mean reversion strategy using RSI and Bollinger Bands"""
        try:
            # Calculate indicators
            rsi = self._calculate_rsi(data)
            bb = self._calculate_bollinger_bands(data)
            current_price = data['Close'].iloc[-1]
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            if (rsi.iloc[-1] < 30 and 
                current_price <= bb['lower'].iloc[-1] * 1.01):  # Within 1% of lower band
                signal = 'BUY'
                confidence = 0.8
                stop_loss = current_price * 0.97  # 3% stop loss
                take_profit = bb['middle'].iloc[-1]  # Target middle band
                
            elif (rsi.iloc[-1] > 70 and 
                  current_price >= bb['upper'].iloc[-1] * 0.99):  # Within 1% of upper band
                signal = 'SELL'
                confidence = 0.8
                stop_loss = current_price * 1.03  # 3% stop loss
                take_profit = bb['middle'].iloc[-1]  # Target middle band
            
            return {
                'strategy': 'Mean Reversion',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'RSI': rsi.iloc[-1], 'BB_Position': current_price / bb['middle'].iloc[-1]}
            }
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {str(e)}")
            return {'strategy': 'Mean Reversion', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _momentum_breakout_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Momentum breakout strategy using volume and price action"""
        try:
            # Calculate 20-period high/low and volume metrics
            high_20 = data['High'].rolling(window=20).max()
            low_20 = data['Low'].rolling(window=20).min()
            avg_volume = data['Volume'].rolling(window=20).mean()
            
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Bullish breakout
            if (current_price > high_20.iloc[-2] and  # Breaking 20-day high
                current_volume > avg_volume.iloc[-1] * 1.5):  # High volume
                signal = 'BUY'
                confidence = 0.75
                stop_loss = low_20.iloc[-1]
                take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 R/R
                
            # Bearish breakdown
            elif (current_price < low_20.iloc[-2] and  # Breaking 20-day low
                  current_volume > avg_volume.iloc[-1] * 1.5):  # High volume
                signal = 'SELL'
                confidence = 0.75
                stop_loss = high_20.iloc[-1]
                take_profit = current_price - (stop_loss - current_price) * 2  # 2:1 R/R
            
            return {
                'strategy': 'Momentum Breakout',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'Volume_Ratio': current_volume / avg_volume.iloc[-1], 'Price_Position': current_price / high_20.iloc[-1]}
            }
        except Exception as e:
            logger.error(f"Error in momentum breakout strategy: {str(e)}")
            return {'strategy': 'Momentum Breakout', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _trend_following_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Trend following strategy using moving averages and MACD"""
        try:
            # Calculate moving averages
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            ema_50 = data['Close'].ewm(span=50).mean()
            
            # MACD
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            
            current_price = data['Close'].iloc[-1]
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Bullish trend
            if (current_price > ema_50.iloc[-1] and
                ema_12.iloc[-1] > ema_26.iloc[-1] and
                macd_line.iloc[-1] > signal_line.iloc[-1]):
                signal = 'BUY'
                confidence = 0.7
                stop_loss = ema_26.iloc[-1]
                take_profit = current_price + (current_price - stop_loss) * 1.5
                
            # Bearish trend
            elif (current_price < ema_50.iloc[-1] and
                  ema_12.iloc[-1] < ema_26.iloc[-1] and
                  macd_line.iloc[-1] < signal_line.iloc[-1]):
                signal = 'SELL'
                confidence = 0.7
                stop_loss = ema_26.iloc[-1]
                take_profit = current_price - (stop_loss - current_price) * 1.5
            
            return {
                'strategy': 'Trend Following',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'EMA_12': ema_12.iloc[-1], 'EMA_26': ema_26.iloc[-1], 'MACD': macd_line.iloc[-1]}
            }
        except Exception as e:
            logger.error(f"Error in trend following strategy: {str(e)}")
            return {'strategy': 'Trend Following', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _swing_trading_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Swing trading strategy using support/resistance and momentum"""
        try:
            # Find swing highs and lows
            swing_high = data['High'].rolling(window=5, center=True).max()
            swing_low = data['Low'].rolling(window=5, center=True).min()
            
            # Recent support and resistance levels
            resistance_level = swing_high.dropna().iloc[-5:].max()
            support_level = swing_low.dropna().iloc[-5:].min()
            
            current_price = data['Close'].iloc[-1]
            rsi = self._calculate_rsi(data)
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Buy at support with oversold RSI
            if (current_price <= support_level * 1.02 and  # Near support
                rsi.iloc[-1] < 40):  # Oversold
                signal = 'BUY'
                confidence = 0.75
                stop_loss = support_level * 0.98
                take_profit = resistance_level
                
            # Sell at resistance with overbought RSI
            elif (current_price >= resistance_level * 0.98 and  # Near resistance
                  rsi.iloc[-1] > 60):  # Overbought
                signal = 'SELL'
                confidence = 0.75
                stop_loss = resistance_level * 1.02
                take_profit = support_level
            
            return {
                'strategy': 'Swing Trading',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'Support': support_level, 'Resistance': resistance_level, 'RSI': rsi.iloc[-1]}
            }
        except Exception as e:
            logger.error(f"Error in swing trading strategy: {str(e)}")
            return {'strategy': 'Swing Trading', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _scalping_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Scalping strategy for very short-term trades"""
        try:
            if len(data) < 20:
                return {'strategy': 'Scalping', 'signal': 'NEUTRAL', 'confidence': 0.0}
            
            # Use short-period indicators
            ema_5 = data['Close'].ewm(span=5).mean()
            ema_10 = data['Close'].ewm(span=10).mean()
            rsi_5 = self._calculate_rsi(data, period=5)
            
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=10).mean().iloc[-1]
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Quick bullish scalp
            if (ema_5.iloc[-1] > ema_10.iloc[-1] and
                rsi_5.iloc[-1] > 50 and rsi_5.iloc[-1] < 70 and
                current_volume > avg_volume * 1.2):
                signal = 'BUY'
                confidence = 0.6
                stop_loss = current_price * 0.999  # Very tight stop
                take_profit = current_price * 1.002  # Small target
                
            # Quick bearish scalp
            elif (ema_5.iloc[-1] < ema_10.iloc[-1] and
                  rsi_5.iloc[-1] < 50 and rsi_5.iloc[-1] > 30 and
                  current_volume > avg_volume * 1.2):
                signal = 'SELL'
                confidence = 0.6
                stop_loss = current_price * 1.001  # Very tight stop
                take_profit = current_price * 0.998  # Small target
            
            return {
                'strategy': 'Scalping',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'EMA_5': ema_5.iloc[-1], 'EMA_10': ema_10.iloc[-1], 'RSI_5': rsi_5.iloc[-1]}
            }
        except Exception as e:
            logger.error(f"Error in scalping strategy: {str(e)}")
            return {'strategy': 'Scalping', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _gap_trading_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Gap trading strategy for market open opportunities"""
        try:
            if len(data) < 2:
                return {'strategy': 'Gap Trading', 'signal': 'NEUTRAL', 'confidence': 0.0}
            
            prev_close = data['Close'].iloc[-2]
            current_open = data['Open'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            gap_percent = (current_open - prev_close) / prev_close * 100
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Gap up fade
            if gap_percent > 2:  # Gap up > 2%
                signal = 'SELL'
                confidence = 0.65
                stop_loss = current_open * 1.01
                take_profit = prev_close  # Target gap fill
                
            # Gap down bounce
            elif gap_percent < -2:  # Gap down > 2%
                signal = 'BUY'
                confidence = 0.65
                stop_loss = current_open * 0.99
                take_profit = prev_close  # Target gap fill
            
            return {
                'strategy': 'Gap Trading',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'Gap_Percent': gap_percent, 'Prev_Close': prev_close, 'Current_Open': current_open}
            }
        except Exception as e:
            logger.error(f"Error in gap trading strategy: {str(e)}")
            return {'strategy': 'Gap Trading', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _volume_spike_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Volume spike strategy for unusual activity"""
        try:
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            if volume_ratio > 3:  # Volume spike > 3x average
                if price_change > 1:  # Positive price movement
                    signal = 'BUY'
                    confidence = 0.7
                    stop_loss = current_price * 0.97
                    take_profit = current_price * 1.06
                elif price_change < -1:  # Negative price movement
                    signal = 'SELL'
                    confidence = 0.7
                    stop_loss = current_price * 1.03
                    take_profit = current_price * 0.94
            
            return {
                'strategy': 'Volume Spike',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'Volume_Ratio': volume_ratio, 'Price_Change': price_change}
            }
        except Exception as e:
            logger.error(f"Error in volume spike strategy: {str(e)}")
            return {'strategy': 'Volume Spike', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _support_resistance_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Support and resistance based strategy"""
        try:
            # Calculate support and resistance levels
            highs = data['High'].rolling(window=10, center=True).max()
            lows = data['Low'].rolling(window=10, center=True).min()
            
            resistance_levels = highs.dropna().tail(10).unique()
            support_levels = lows.dropna().tail(10).unique()
            
            current_price = data['Close'].iloc[-1]
            
            # Find nearest support and resistance
            resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            
            signal = 'NEUTRAL'
            confidence = 0.0
            entry_price = current_price
            stop_loss = 0.0
            take_profit = 0.0
            
            # Near support - potential buy
            if current_price <= support * 1.01:
                signal = 'BUY'
                confidence = 0.7
                stop_loss = support * 0.98
                take_profit = resistance
                
            # Near resistance - potential sell
            elif current_price >= resistance * 0.99:
                signal = 'SELL'
                confidence = 0.7
                stop_loss = resistance * 1.02
                take_profit = support
            
            return {
                'strategy': 'Support Resistance',
                'signal': signal,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 0,
                'indicators': {'Support': support, 'Resistance': resistance, 'Distance_to_Support': (current_price - support) / support * 100}
            }
        except Exception as e:
            logger.error(f"Error in support resistance strategy: {str(e)}")
            return {'strategy': 'Support Resistance', 'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std_dev = data['Close'].rolling(window=period).std()
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    def run_all_strategies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run all trading strategies and compile results"""
        try:
            strategy_results = []
            
            for strategy_name, strategy_func in self.strategies.items():
                result = strategy_func(data)
                if result.get('confidence', 0) > 0:
                    strategy_results.append(result)
            
            # Sort by confidence
            strategy_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Calculate consensus
            buy_signals = len([s for s in strategy_results if s.get('signal') == 'BUY'])
            sell_signals = len([s for s in strategy_results if s.get('signal') == 'SELL'])
            total_signals = len(strategy_results)
            
            consensus_signal = 'NEUTRAL'
            consensus_confidence = 0.0
            
            if buy_signals > sell_signals and buy_signals >= 2:
                consensus_signal = 'BUY'
                consensus_confidence = buy_signals / total_signals if total_signals > 0 else 0
            elif sell_signals > buy_signals and sell_signals >= 2:
                consensus_signal = 'SELL' 
                consensus_confidence = sell_signals / total_signals if total_signals > 0 else 0
            
            return {
                'consensus_signal': consensus_signal,
                'consensus_confidence': consensus_confidence,
                'strategy_results': strategy_results,
                'total_strategies': len(self.strategies),
                'active_strategies': len(strategy_results)
            }
            
        except Exception as e:
            logger.error(f"Error running strategies: {str(e)}")
            return {'consensus_signal': 'NEUTRAL', 'consensus_confidence': 0.0, 'strategy_results': []}