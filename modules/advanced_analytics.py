import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced trading analytics and signals for professional trading"""
    
    def __init__(self):
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = data['Close'].ewm(span=self.macd_fast).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = data['Close'].rolling(window=period).mean()
        std_dev = data['Close'].rolling(window=period).std()
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        
        k_percent = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift())
        low_close_prev = abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Calculate Volume Profile - shows volume at different price levels"""
        try:
            price_range = data['High'].max() - data['Low'].min()
            price_levels = np.linspace(data['Low'].min(), data['High'].max(), bins)
            volume_profile = np.zeros(bins-1)
            
            for i in range(len(data)):
                row = data.iloc[i]
                typical_price = (row['High'] + row['Low'] + row['Close']) / 3
                bin_idx = np.digitize(typical_price, price_levels) - 1
                if 0 <= bin_idx < len(volume_profile):
                    volume_profile[bin_idx] += row['Volume']
            
            # Find Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_levels[poc_idx] + price_levels[poc_idx + 1]) / 2
            
            return {
                'volume_profile': volume_profile,
                'price_levels': price_levels[:-1],
                'poc_price': poc_price,
                'total_volume': data['Volume'].sum()
            }
        except Exception:
            return {'volume_profile': [], 'price_levels': [], 'poc_price': 0, 'total_volume': 0}
    
    def calculate_fibonacci_levels(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            recent_data = data.tail(lookback)
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            diff = high - low
            
            return {
                'high': high,
                'low': low,
                'fib_23.6': high - 0.236 * diff,
                'fib_38.2': high - 0.382 * diff,
                'fib_50.0': high - 0.500 * diff,
                'fib_61.8': high - 0.618 * diff,
                'fib_78.6': high - 0.786 * diff,
                'fib_100.0': low
            }
        except Exception:
            return {}
    
    def calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components"""
        try:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = data['High'].rolling(window=9).max()
            period9_low = data['Low'].rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = data['High'].rolling(window=26).max()
            period26_low = data['Low'].rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A: (Tenkan-sen + Kijun-sen)/2, projected 26 periods ahead
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B: (52-period high + 52-period low)/2, projected 26 periods ahead
            period52_high = data['High'].rolling(window=52).max()
            period52_low = data['Low'].rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span: Close price projected 26 periods back
            chikou_span = data['Close'].shift(-26)
            
            return {
                'tenkan_sen': tenkan_sen,
                'kijun_sen': kijun_sen,
                'senkou_span_a': senkou_span_a,
                'senkou_span_b': senkou_span_b,
                'chikou_span': chikou_span
            }
        except Exception:
            return {}
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
            return williams_r
        except Exception:
            return pd.Series()
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            return cci
        except Exception:
            return pd.Series()
    
    def calculate_parabolic_sar(self, data: pd.DataFrame, af_start=0.02, af_increment=0.02, af_max=0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            psar = np.zeros(len(data))
            trend = np.zeros(len(data))
            af = af_start
            ep = 0
            
            # Initialize
            psar[0] = low[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            
            for i in range(1, len(data)):
                if trend[i-1] == 1:  # Uptrend
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])
                    
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
                    
                    if low[i] < psar[i]:
                        trend[i] = -1
                        psar[i] = ep
                        af = af_start
                        ep = low[i]
                    else:
                        trend[i] = 1
                else:  # Downtrend
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])
                    
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
                    
                    if high[i] > psar[i]:
                        trend[i] = 1
                        psar[i] = ep
                        af = af_start
                        ep = high[i]
                    else:
                        trend[i] = -1
            
            return pd.Series(psar, index=data.index)
        except Exception:
            return pd.Series()
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            return pd.Series(obv, index=data.index)
        except Exception:
            return pd.Series()
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
            return vwap
        except Exception:
            return pd.Series()
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points for current trading session"""
        try:
            # Use previous day's data for pivot calculation
            prev_high = data['High'].iloc[-2] if len(data) > 1 else data['High'].iloc[-1]
            prev_low = data['Low'].iloc[-2] if len(data) > 1 else data['Low'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            
            return {
                'pivot': pivot,
                'r1': 2 * pivot - prev_low,
                'r2': pivot + (prev_high - prev_low),
                'r3': prev_high + 2 * (pivot - prev_low),
                's1': 2 * pivot - prev_high,
                's2': pivot - (prev_high - prev_low),
                's3': prev_low - 2 * (prev_high - pivot)
            }
        except Exception:
            return {}
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for trading intelligence"""
        try:
            indicators = {}
            
            # RSI
            rsi = self.calculate_rsi(data)
            if not rsi.empty:
                indicators['rsi'] = {
                    'current': rsi.iloc[-1],
                    'series': rsi,
                    'overbought': rsi.iloc[-1] > 70,
                    'oversold': rsi.iloc[-1] < 30
                }
            
            # MACD
            macd_data = self.calculate_macd(data)
            if macd_data and 'macd' in macd_data:
                indicators['macd'] = {
                    'macd': macd_data['macd'].iloc[-1],
                    'signal': macd_data['signal'].iloc[-1],
                    'histogram': macd_data['histogram'].iloc[-1],
                    'bullish_crossover': (macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1] and 
                                        macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]),
                    'bearish_crossover': (macd_data['macd'].iloc[-1] < macd_data['signal'].iloc[-1] and 
                                        macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2])
                }
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(data)
            if bb_data and 'upper' in bb_data:
                current_price = data['Close'].iloc[-1]
                indicators['bollinger_bands'] = {
                    'upper': bb_data['upper'].iloc[-1],
                    'middle': bb_data['middle'].iloc[-1],
                    'lower': bb_data['lower'].iloc[-1],
                    'current_price': current_price,
                    'price_position': 'upper' if current_price > bb_data['upper'].iloc[-1] else 
                                   'lower' if current_price < bb_data['lower'].iloc[-1] else 'middle',
                    'bandwidth': (bb_data['upper'].iloc[-1] - bb_data['lower'].iloc[-1]) / bb_data['middle'].iloc[-1]
                }
            
            # Stochastic
            stoch_data = self.calculate_stochastic(data)
            if stoch_data and 'k' in stoch_data:
                indicators['stochastic'] = {
                    'k': stoch_data['k'].iloc[-1],
                    'd': stoch_data['d'].iloc[-1],
                    'overbought': stoch_data['k'].iloc[-1] > 80,
                    'oversold': stoch_data['k'].iloc[-1] < 20
                }
            
            # ATR
            atr = self.calculate_atr(data)
            if not atr.empty:
                indicators['atr'] = {
                    'current': atr.iloc[-1],
                    'series': atr,
                    'volatility_level': 'high' if atr.iloc[-1] > atr.mean() * 1.5 else 
                                      'low' if atr.iloc[-1] < atr.mean() * 0.5 else 'normal'
                }
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            indicators['volume'] = {
                'current': current_volume,
                'average': avg_volume,
                'relative': current_volume / avg_volume if avg_volume > 0 else 1,
                'high_volume': current_volume > avg_volume * 1.5
            }
            
            # Support/Resistance levels
            support_resistance = self.calculate_support_resistance_levels(data)
            indicators['support_resistance'] = support_resistance
            
            # Pivot Points
            pivots = self.calculate_pivot_points(data)
            indicators['pivot_points'] = pivots
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def generate_basic_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic trading signals for compatibility"""
        try:
            signals = {
                'overall_signal': 'NEUTRAL',
                'confidence': 0.0,
                'signals': [],
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'risk_reward_ratio': 0.0
            }
            
            current_price = data['Close'].iloc[-1]
            
            # RSI Analysis
            rsi = self.calculate_rsi(data)
            if not rsi.empty:
                rsi_current = rsi.iloc[-1]
                if rsi_current < 30:
                    signals['signals'].append({'indicator': 'RSI', 'signal': 'BUY', 'strength': 'Strong', 'value': rsi_current})
                elif rsi_current > 70:
                    signals['signals'].append({'indicator': 'RSI', 'signal': 'SELL', 'strength': 'Strong', 'value': rsi_current})
                else:
                    signals['signals'].append({'indicator': 'RSI', 'signal': 'NEUTRAL', 'strength': 'Weak', 'value': rsi_current})
            
            # MACD Analysis
            macd_data = self.calculate_macd(data)
            if macd_data and 'macd' in macd_data:
                macd_line = macd_data['macd'].iloc[-1]
                signal_line = macd_data['signal'].iloc[-1]
                
                if macd_line > signal_line and macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]:
                    signals['signals'].append({'indicator': 'MACD', 'signal': 'BUY', 'strength': 'Medium', 'value': macd_line - signal_line})
                elif macd_line < signal_line and macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2]:
                    signals['signals'].append({'indicator': 'MACD', 'signal': 'SELL', 'strength': 'Medium', 'value': macd_line - signal_line})
            
            # Bollinger Bands Analysis
            bb_data = self.calculate_bollinger_bands(data)
            if bb_data and 'upper' in bb_data:
                if current_price <= bb_data['lower'].iloc[-1]:
                    signals['signals'].append({'indicator': 'Bollinger Bands', 'signal': 'BUY', 'strength': 'Medium', 'value': current_price})
                elif current_price >= bb_data['upper'].iloc[-1]:
                    signals['signals'].append({'indicator': 'Bollinger Bands', 'signal': 'SELL', 'strength': 'Medium', 'value': current_price})
            
            # Volume Analysis
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                signals['signals'].append({'indicator': 'Volume', 'signal': 'BREAKOUT', 'strength': 'Strong', 'value': current_volume / avg_volume})
            
            # Calculate overall signal
            buy_signals = len([s for s in signals['signals'] if s['signal'] == 'BUY'])
            sell_signals = len([s for s in signals['signals'] if s['signal'] == 'SELL'])
            
            if buy_signals > sell_signals and buy_signals >= 2:
                signals['overall_signal'] = 'BUY'
                signals['confidence'] = min(0.9, buy_signals * 0.2)
                
                # Set entry and exit levels
                atr = self.calculate_atr(data).iloc[-1] if not self.calculate_atr(data).empty else current_price * 0.02
                signals['entry_price'] = current_price
                signals['stop_loss'] = current_price - (2 * atr)
                signals['take_profit'] = current_price + (3 * atr)
                signals['risk_reward_ratio'] = 1.5
                
            elif sell_signals > buy_signals and sell_signals >= 2:
                signals['overall_signal'] = 'SELL'
                signals['confidence'] = min(0.9, sell_signals * 0.2)
                
                # Set entry and exit levels
                atr = self.calculate_atr(data).iloc[-1] if not self.calculate_atr(data).empty else current_price * 0.02
                signals['entry_price'] = current_price
                signals['stop_loss'] = current_price + (2 * atr)
                signals['take_profit'] = current_price - (3 * atr)
                signals['risk_reward_ratio'] = 1.5
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return {'overall_signal': 'NEUTRAL', 'confidence': 0.0, 'signals': []}
    
    def detect_divergences(self, data: pd.DataFrame, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect price-indicator divergences"""
        divergences = []
        
        # Calculate RSI for divergence detection
        rsi = self.calculate_rsi(data)
        
        for i in range(len(data) - 20, len(data)):
            if i < 20:
                continue
                
            # Look for bullish divergence (lower lows in price, higher lows in RSI)
            price_window = data['Close'].iloc[i-10:i+1]
            rsi_window = rsi.iloc[i-10:i+1]
            
            if len(price_window) >= 2 and len(rsi_window) >= 2:
                price_min_idx = price_window.idxmin()
                rsi_min_idx = rsi_window.idxmin()
                
                # Check for bullish divergence
                if (price_window.iloc[-1] < price_window.iloc[0] and 
                    rsi_window.iloc[-1] > rsi_window.iloc[0]):
                    
                    divergences.append({
                        'type': 'Bullish Divergence',
                        'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                        'price': data['Close'].iloc[i],
                        'rsi': rsi.iloc[i],
                        'signal': 'Potential Bullish Reversal',
                        'confidence': 0.7
                    })
                
                # Check for bearish divergence
                elif (price_window.iloc[-1] > price_window.iloc[0] and 
                      rsi_window.iloc[-1] < rsi_window.iloc[0]):
                    
                    divergences.append({
                        'type': 'Bearish Divergence',
                        'date': data.index[i].strftime('%Y-%m-%d %H:%M'),
                        'price': data['Close'].iloc[i],
                        'rsi': rsi.iloc[i],
                        'signal': 'Potential Bearish Reversal',
                        'confidence': 0.7
                    })
        
        return divergences
    
    def calculate_support_resistance_levels(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Calculate dynamic support and resistance levels"""
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(data['High'].iloc[i])
            
            if data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(data['Low'].iloc[i])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[-5:]
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Calculate volume profile for price levels"""
        price_range = data['High'].max() - data['Low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            level_low = data['Low'].min() + (i * bin_size)
            level_high = level_low + bin_size
            
            # Find candles that traded in this price range
            mask = ((data['Low'] <= level_high) & (data['High'] >= level_low))
            volume_in_range = data.loc[mask, 'Volume'].sum()
            
            mid_price = (level_low + level_high) / 2
            volume_profile[mid_price] = volume_in_range
        
        # Find Point of Control (highest volume level)
        poc = max(volume_profile, key=volume_profile.get)
        
        return {
            'profile': volume_profile,
            'poc': poc,
            'total_volume': data['Volume'].sum()
        }
    
    def generate_trading_signals(self, data: pd.DataFrame, patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        signal_list = []
        if patterns is None:
            patterns = []
        
        # Calculate indicators
        rsi = self.calculate_rsi(data)
        macd = self.calculate_macd(data)
        bb = self.calculate_bollinger_bands(data)
        stoch = self.calculate_stochastic(data)
        atr = self.calculate_atr(data)
        
        current_idx = len(data) - 1
        current_price = data['Close'].iloc[current_idx]
        
        # RSI signals
        try:
            if len(rsi) > current_idx:
                if rsi.iloc[current_idx] < 30:
                    signal_list.append({
                        'type': 'RSI Oversold',
                        'signal': 'Potential Buy Signal',
                        'value': f"RSI: {rsi.iloc[current_idx]:.1f}",
                        'strength': 'Strong' if rsi.iloc[current_idx] < 20 else 'Moderate',
                        'timeframe': 'Short-term'
                    })
                elif rsi.iloc[current_idx] > 70:
                    signal_list.append({
                        'type': 'RSI Overbought',
                        'signal': 'Potential Sell Signal',
                        'value': f"RSI: {rsi.iloc[current_idx]:.1f}",
                        'strength': 'Strong' if rsi.iloc[current_idx] > 80 else 'Moderate',
                        'timeframe': 'Short-term'
                    })
        except Exception as e:
            logger.warning(f"RSI signal calculation error: {str(e)}")
        
        # MACD signals
        try:
            if (macd and 'macd' in macd and 'signal' in macd and 
                len(macd['macd']) > current_idx and len(macd['signal']) > current_idx and
                current_idx > 0):
                
                if (macd['macd'].iloc[current_idx] > macd['signal'].iloc[current_idx] and 
                    macd['macd'].iloc[current_idx-1] <= macd['signal'].iloc[current_idx-1]):
                    signal_list.append({
                        'type': 'MACD Bullish Crossover',
                        'signal': 'Buy Signal',
                        'value': f"MACD: {macd['macd'].iloc[current_idx]:.4f}",
                        'strength': 'Strong',
                        'timeframe': 'Medium-term'
                    })
                elif (macd['macd'].iloc[current_idx] < macd['signal'].iloc[current_idx] and 
                      macd['macd'].iloc[current_idx-1] >= macd['signal'].iloc[current_idx-1]):
                    signal_list.append({
                        'type': 'MACD Bearish Crossover',
                        'signal': 'Sell Signal',
                        'value': f"MACD: {macd['macd'].iloc[current_idx]:.4f}",
                        'strength': 'Strong',
                        'timeframe': 'Medium-term'
                    })
        except Exception as e:
            logger.warning(f"MACD signal calculation error: {str(e)}")
        
        # Bollinger Bands signals
        try:
            if (bb and 'upper' in bb and 'lower' in bb and 
                len(bb['upper']) > current_idx and len(bb['lower']) > current_idx):
                
                if current_price <= bb['lower'].iloc[current_idx]:
                    signal_list.append({
                        'type': 'Bollinger Band Oversold',
                        'signal': 'Potential Buy Signal',
                        'value': f"Price: ${current_price:.2f}, Lower Band: ${bb['lower'].iloc[current_idx]:.2f}",
                        'strength': 'Moderate',
                        'timeframe': 'Short-term'
                    })
                elif current_price >= bb['upper'].iloc[current_idx]:
                    signal_list.append({
                        'type': 'Bollinger Band Overbought',
                        'signal': 'Potential Sell Signal',
                        'value': f"Price: ${current_price:.2f}, Upper Band: ${bb['upper'].iloc[current_idx]:.2f}",
                        'strength': 'Moderate',
                        'timeframe': 'Short-term'
                    })
        except Exception as e:
            logger.warning(f"Bollinger Bands signal calculation error: {str(e)}")
        
        # Return in expected format
        return {
            'signals': signal_list,
            'overall_signal': 'BUY' if len([s for s in signal_list if 'Buy' in s.get('signal', '')]) > len([s for s in signal_list if 'Sell' in s.get('signal', '')]) else 'SELL' if len([s for s in signal_list if 'Sell' in s.get('signal', '')]) > 0 else 'NEUTRAL',
            'confidence': min(1.0, len(signal_list) * 0.2)
        }
    
    def calculate_risk_metrics(self, data: pd.DataFrame, position_size: float = 1000) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        returns = data['Close'].pct_change().dropna()
        atr = self.calculate_atr(data)
        current_price = data['Close'].iloc[-1]
        current_atr = atr.iloc[-1]
        
        # Position sizing based on ATR
        risk_per_share = current_atr * 2  # 2x ATR stop loss
        shares = position_size / risk_per_share if risk_per_share > 0 else 0
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        return {
            'position_size_shares': int(shares),
            'risk_per_share': risk_per_share,
            'stop_loss_level': current_price - risk_per_share,
            'take_profit_level': current_price + (risk_per_share * 2),  # 2:1 risk/reward
            'daily_volatility': volatility,
            'max_drawdown': max_drawdown,
            'current_atr': current_atr
        }
    
    def backtest_patterns(self, data: pd.DataFrame, patterns: List[Dict[str, Any]], 
                         holding_period: int = 5) -> Dict[str, Any]:
        """Simple backtest of pattern performance"""
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            logger.error(f"backtest_patterns received {type(data)} instead of DataFrame")
            return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0, 'pattern_performance': {}}
        
        if not isinstance(patterns, list):
            logger.error(f"backtest_patterns received {type(patterns)} instead of list for patterns")
            return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0, 'pattern_performance': {}}
            
        results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'pattern_performance': {}
        }
        
        for pattern in patterns:
            # Validate pattern structure
            if not isinstance(pattern, dict):
                logger.error(f"Pattern is {type(pattern)} instead of dict: {pattern}")
                continue
            
            if 'type' not in pattern or 'start_idx' not in pattern:
                logger.error(f"Pattern missing required keys: {pattern}")
                continue
                
            pattern_type = pattern['type']
            start_idx = pattern['start_idx']
            
            # Skip if not enough data for holding period
            if start_idx + holding_period >= len(data):
                continue
            
            entry_price = data['Close'].iloc[start_idx]
            exit_price = data['Close'].iloc[start_idx + holding_period]
            
            # Determine trade direction based on signal
            signal = pattern.get('signal', '')
            if 'Bullish' in signal:
                trade_return = (exit_price - entry_price) / entry_price
            elif 'Bearish' in signal:
                trade_return = (entry_price - exit_price) / entry_price
            else:
                continue  # Skip neutral signals
            
            results['total_trades'] += 1
            results['total_return'] += trade_return
            
            if trade_return > 0:
                results['winning_trades'] += 1
            else:
                results['losing_trades'] += 1
            
            # Track by pattern type
            if pattern_type not in results['pattern_performance']:
                results['pattern_performance'][pattern_type] = {
                    'trades': 0,
                    'wins': 0,
                    'total_return': 0.0
                }
            
            results['pattern_performance'][pattern_type]['trades'] += 1
            results['pattern_performance'][pattern_type]['total_return'] += trade_return
            if trade_return > 0:
                results['pattern_performance'][pattern_type]['wins'] += 1
        
        # Calculate metrics
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            results['avg_return'] = results['total_return'] / results['total_trades']
        
        return results