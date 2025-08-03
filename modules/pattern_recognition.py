import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PatternRecognition:
    """Identifies candlestick patterns in OHLCV data"""
    
    def __init__(self):
        self.min_confidence = 0.6
        self.patterns = {
            'Hammer': self._identify_hammer,
            'Inverted Hammer': self._identify_inverted_hammer,
            'Shooting Star': self._identify_shooting_star,
            'Hanging Man': self._identify_hanging_man,
            'Doji': self._identify_doji,
            'Bullish Engulfing': self._identify_bullish_engulfing,
            'Bearish Engulfing': self._identify_bearish_engulfing,
            'Morning Star': self._identify_morning_star,
            'Evening Star': self._identify_evening_star,
            'Three White Soldiers': self._identify_three_white_soldiers,
            'Three Black Crows': self._identify_three_black_crows,
            'Piercing Pattern': self._identify_piercing_pattern,
            'Dark Cloud Cover': self._identify_dark_cloud_cover,

        }
    
    def _calculate_shadows(self, row):
        """Calculate upper and lower shadows for a candlestick"""
        try:
            body_top = max(row['Open'], row['Close'])
            body_bottom = min(row['Open'], row['Close'])
            upper_shadow = row['High'] - body_top
            lower_shadow = body_bottom - row['Low']
            return upper_shadow, lower_shadow
        except Exception as e:
            logger.error(f"Error calculating shadows: {str(e)}")
            return 0, 0
    
    def analyze_patterns(self, data: pd.DataFrame, pattern_types: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze candlestick patterns in the provided data
        
        Args:
            data: DataFrame with OHLCV data
            pattern_types: List of pattern types to analyze
            
        Returns:
            List of detected patterns with metadata
        """
        if data is None or data.empty:
            return []
        
        patterns_found = []
        
        try:
            for pattern_type in pattern_types:
                if pattern_type in self.patterns:
                    pattern_func = self.patterns[pattern_type]
                    detected_patterns = pattern_func(data)
                    patterns_found.extend(detected_patterns)
            
            # Sort patterns by date
            patterns_found.sort(key=lambda x: x['date'])
            
            logger.info(f"Found {len(patterns_found)} patterns")
            return patterns_found
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return []
    
    def _identify_hammer(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Hammer patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            # Hammer criteria - calculate shadows using utility function
            body_size = abs(current['Close'] - current['Open'])
            upper_shadow, lower_shadow = self._calculate_shadows(current)
            total_range = current['High'] - current['Low']
            
            if (total_range > 0 and
                lower_shadow >= 2 * body_size and  # Long lower shadow
                upper_shadow <= 0.1 * total_range and  # Small upper shadow
                body_size >= 0.1 * total_range):  # Reasonable body size
                
                confidence = min(0.9, (lower_shadow / body_size) / 3)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Hammer',
                        'date': current.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i,
                        'end_idx': i,
                        'signal': 'Bullish Reversal'
                    })
        
        return patterns
    
    def _identify_inverted_hammer(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Inverted Hammer patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            body_size = abs(current['Close'] - current['Open'])
            upper_shadow, lower_shadow = self._calculate_shadows(current)
            total_range = current['High'] - current['Low']
            
            if (total_range > 0 and
                upper_shadow >= 2 * body_size and  # Long upper shadow
                lower_shadow <= 0.1 * total_range and  # Small lower shadow
                body_size >= 0.1 * total_range):
                
                confidence = min(0.9, (upper_shadow / body_size) / 3)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Inverted Hammer',
                        'date': current.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i,
                        'end_idx': i,
                        'signal': 'Potential Bullish Reversal'
                    })
        
        return patterns
    
    def _identify_shooting_star(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Shooting Star patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            body_size = abs(current['Close'] - current['Open'])
            upper_shadow, lower_shadow = self._calculate_shadows(current)
            total_range = current['High'] - current['Low']
            
            # Shooting star appears at top of uptrend
            if i >= 5:  # Need some history to determine trend
                recent_trend = data.iloc[i-5:i]['Close'].mean() < current['Close']
                
                if (total_range > 0 and recent_trend and
                    upper_shadow >= 2 * body_size and
                    lower_shadow <= 0.1 * total_range and
                    body_size >= 0.1 * total_range):
                    
                    confidence = min(0.9, (upper_shadow / body_size) / 3)
                    
                    if confidence >= self.min_confidence:
                        patterns.append({
                            'type': 'Shooting Star',
                            'date': current.name.strftime('%Y-%m-%d %H:%M'),
                            'confidence': confidence,
                            'start_idx': i,
                            'end_idx': i,
                            'signal': 'Bearish Reversal'
                        })
        
        return patterns
    
    def _identify_hanging_man(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Hanging Man patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            
            body_size = abs(current['Close'] - current['Open'])
            upper_shadow, lower_shadow = self._calculate_shadows(current)
            total_range = current['High'] - current['Low']
            
            # Hanging man appears at top of uptrend
            if i >= 5:
                recent_trend = data.iloc[i-5:i]['Close'].mean() < current['Close']
                
                if (total_range > 0 and recent_trend and
                    lower_shadow >= 2 * body_size and
                    upper_shadow <= 0.1 * total_range and
                    body_size >= 0.1 * total_range):
                    
                    confidence = min(0.9, (lower_shadow / body_size) / 3)
                    
                    if confidence >= self.min_confidence:
                        patterns.append({
                            'type': 'Hanging Man',
                            'date': current.name.strftime('%Y-%m-%d %H:%M'),
                            'confidence': confidence,
                            'start_idx': i,
                            'end_idx': i,
                            'signal': 'Bearish Reversal'
                        })
        
        return patterns
    
    def _identify_doji(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Doji patterns"""
        patterns = []
        
        for i in range(len(data)):
            current = data.iloc[i]
            
            body_size = abs(current['Close'] - current['Open'])
            total_range = current['High'] - current['Low']
            
            if total_range > 0:
                # Doji has very small body relative to range
                body_ratio = body_size / total_range
                
                if body_ratio <= 0.1:  # Body is less than 10% of total range
                    confidence = 1.0 - (body_ratio * 10)  # Higher confidence for smaller bodies
                    
                    if confidence >= self.min_confidence:
                        patterns.append({
                            'type': 'Doji',
                            'date': current.name.strftime('%Y-%m-%d %H:%M'),
                            'confidence': confidence,
                            'start_idx': i,
                            'end_idx': i,
                            'signal': 'Market Indecision'
                        })
        
        return patterns
    
    def _identify_bullish_engulfing(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Bullish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Previous candle is bearish, current is bullish
            if (previous['Close'] < previous['Open'] and
                current['Close'] > current['Open'] and
                current['Open'] < previous['Close'] and
                current['Close'] > previous['Open']):
                
                # Calculate engulfing strength
                prev_body = abs(previous['Close'] - previous['Open'])
                curr_body = abs(current['Close'] - current['Open'])
                
                if curr_body > prev_body:
                    confidence = min(0.95, curr_body / prev_body / 2)
                    
                    if confidence >= self.min_confidence:
                        patterns.append({
                            'type': 'Bullish Engulfing',
                            'date': current.name.strftime('%Y-%m-%d %H:%M'),
                            'confidence': confidence,
                            'start_idx': i-1,
                            'end_idx': i,
                            'signal': 'Bullish Reversal'
                        })
        
        return patterns
    
    def _identify_bearish_engulfing(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Bearish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Previous candle is bullish, current is bearish
            if (previous['Close'] > previous['Open'] and
                current['Close'] < current['Open'] and
                current['Open'] > previous['Close'] and
                current['Close'] < previous['Open']):
                
                prev_body = abs(previous['Close'] - previous['Open'])
                curr_body = abs(current['Close'] - current['Open'])
                
                if curr_body > prev_body:
                    confidence = min(0.95, curr_body / prev_body / 2)
                    
                    if confidence >= self.min_confidence:
                        patterns.append({
                            'type': 'Bearish Engulfing',
                            'date': current.name.strftime('%Y-%m-%d %H:%M'),
                            'confidence': confidence,
                            'start_idx': i-1,
                            'end_idx': i,
                            'signal': 'Bearish Reversal'
                        })
        
        return patterns
    
    def _identify_morning_star(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Morning Star patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # First candle: bearish
            # Second candle: small body (star)
            # Third candle: bullish, closes above first candle's midpoint
            
            first_body = abs(first['Close'] - first['Open'])
            second_body = abs(second['Close'] - second['Open'])
            third_body = abs(third['Close'] - third['Open'])
            
            first_midpoint = (first['Open'] + first['Close']) / 2
            
            if (first['Close'] < first['Open'] and  # First is bearish
                second_body < first_body * 0.5 and  # Second has small body
                third['Close'] > third['Open'] and  # Third is bullish
                third['Close'] > first_midpoint):  # Third closes above first's midpoint
                
                confidence = min(0.9, third_body / first_body)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Morning Star',
                        'date': third.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-2,
                        'end_idx': i,
                        'signal': 'Bullish Reversal'
                    })
        
        return patterns
    
    def _identify_evening_star(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Evening Star patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            first_body = abs(first['Close'] - first['Open'])
            second_body = abs(second['Close'] - second['Open'])
            third_body = abs(third['Close'] - third['Open'])
            
            first_midpoint = (first['Open'] + first['Close']) / 2
            
            if (first['Close'] > first['Open'] and  # First is bullish
                second_body < first_body * 0.5 and  # Second has small body
                third['Close'] < third['Open'] and  # Third is bearish
                third['Close'] < first_midpoint):  # Third closes below first's midpoint
                
                confidence = min(0.9, third_body / first_body)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Evening Star',
                        'date': third.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-2,
                        'end_idx': i,
                        'signal': 'Bearish Reversal'
                    })
        
        return patterns
    
    def _identify_three_white_soldiers(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Three White Soldiers patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # All three candles are bullish and each opens within previous body
            if (first['Close'] > first['Open'] and
                second['Close'] > second['Open'] and
                third['Close'] > third['Open'] and
                second['Open'] > first['Open'] and second['Open'] < first['Close'] and
                third['Open'] > second['Open'] and third['Open'] < second['Close'] and
                third['Close'] > second['Close'] > first['Close']):
                
                # Calculate pattern strength
                first_body = first['Close'] - first['Open']
                second_body = second['Close'] - second['Open']
                third_body = third['Close'] - third['Open']
                
                avg_body = (first_body + second_body + third_body) / 3
                confidence = min(0.9, avg_body / (third['High'] - first['Low']))
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Three White Soldiers',
                        'date': third.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-2,
                        'end_idx': i,
                        'signal': 'Strong Bullish Continuation'
                    })
        
        return patterns
    
    def _identify_three_black_crows(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Three Black Crows patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            first = data.iloc[i-2]
            second = data.iloc[i-1]
            third = data.iloc[i]
            
            # All three candles are bearish and each opens within previous body
            if (first['Close'] < first['Open'] and
                second['Close'] < second['Open'] and
                third['Close'] < third['Open'] and
                second['Open'] < first['Open'] and second['Open'] > first['Close'] and
                third['Open'] < second['Open'] and third['Open'] > second['Close'] and
                third['Close'] < second['Close'] < first['Close']):
                
                first_body = first['Open'] - first['Close']
                second_body = second['Open'] - second['Close']
                third_body = third['Open'] - third['Close']
                
                avg_body = (first_body + second_body + third_body) / 3
                confidence = min(0.9, avg_body / (first['High'] - third['Low']))
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Three Black Crows',
                        'date': third.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-2,
                        'end_idx': i,
                        'signal': 'Strong Bearish Continuation'
                    })
        
        return patterns
    
    def _identify_piercing_pattern(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Piercing Pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            first = data.iloc[i-1]
            second = data.iloc[i]
            
            # First candle bearish, second bullish and closes above first's midpoint
            first_midpoint = (first['Open'] + first['Close']) / 2
            
            if (first['Close'] < first['Open'] and
                second['Close'] > second['Open'] and
                second['Open'] < first['Close'] and
                second['Close'] > first_midpoint and
                second['Close'] < first['Open']):
                
                penetration = (second['Close'] - first['Close']) / (first['Open'] - first['Close'])
                confidence = min(0.9, penetration)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Piercing Pattern',
                        'date': second.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-1,
                        'end_idx': i,
                        'signal': 'Bullish Reversal'
                    })
        
        return patterns
    
    def _identify_dark_cloud_cover(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify Dark Cloud Cover pattern"""
        patterns = []
        
        for i in range(1, len(data)):
            first = data.iloc[i-1]
            second = data.iloc[i]
            
            # First candle bullish, second bearish and closes below first's midpoint
            first_midpoint = (first['Open'] + first['Close']) / 2
            
            if (first['Close'] > first['Open'] and
                second['Close'] < second['Open'] and
                second['Open'] > first['Close'] and
                second['Close'] < first_midpoint and
                second['Close'] > first['Open']):
                
                penetration = (first['Close'] - second['Close']) / (first['Close'] - first['Open'])
                confidence = min(0.9, penetration)
                
                if confidence >= self.min_confidence:
                    patterns.append({
                        'type': 'Dark Cloud Cover',
                        'date': second.name.strftime('%Y-%m-%d %H:%M'),
                        'confidence': confidence,
                        'start_idx': i-1,
                        'end_idx': i,
                        'signal': 'Bearish Reversal'
                    })
        
        return patterns
