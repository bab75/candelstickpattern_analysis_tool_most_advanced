import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class Utils:
    """Utility functions for the candlestick analysis application"""
    
    def __init__(self):
        self.market_timezones = {
            'NYSE': 'America/New_York',
            'NASDAQ': 'America/New_York',
            'LSE': 'Europe/London',
            'TSE': 'Asia/Tokyo',
            'ASX': 'Australia/Sydney'
        }
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market status for a given symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with market status information
        """
        try:
            # Default to NYSE timezone for most US stocks
            market_tz = pytz.timezone('America/New_York')
            now = datetime.now(market_tz)
            
            # Market hours (9:30 AM - 4:00 PM ET)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if it's a weekday
            is_weekday = now.weekday() < 5
            
            # Check if market is currently open
            is_open = (is_weekday and 
                      market_open <= now <= market_close)
            
            # Determine current session
            if is_open:
                session = "Regular Trading Hours"
            elif is_weekday and now < market_open:
                session = "Pre-Market"
            elif is_weekday and now > market_close:
                session = "After-Hours"
            else:
                session = "Market Closed (Weekend)"
            
            # Calculate next market open
            if is_weekday and now < market_open:
                next_open = market_open
            elif is_weekday and now >= market_close:
                next_open = market_open + timedelta(days=1)
            else:
                # Weekend - find next Monday
                days_ahead = 7 - now.weekday()
                if now.weekday() == 6:  # Sunday
                    days_ahead = 1
                next_open = (now + timedelta(days=days_ahead)).replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
            
            return {
                'is_open': is_open,
                'session': session,
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'next_open': next_open.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'timezone': 'America/New_York'
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return {
                'is_open': False,
                'session': 'Unknown',
                'current_time': 'Unknown',
                'next_open': 'Unknown',
                'timezone': 'Unknown'
            }
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Boolean indicating if symbol is valid
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get basic info
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except:
            return False
    
    def format_price(self, price: float, currency: str = 'USD') -> str:
        """Format price with appropriate currency symbol"""
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'C$',
            'AUD': 'A$'
        }
        
        symbol = currency_symbols.get(currency, '$')
        return f"{symbol}{price:.2f}"
    
    def format_volume(self, volume: int) -> str:
        """Format volume with appropriate suffixes"""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return str(volume)
    
    def calculate_returns(self, data: pd.DataFrame, period: int = 1) -> pd.Series:
        """Calculate returns for given period"""
        if 'Close' not in data.columns:
            return pd.Series()
        
        return data['Close'].pct_change(periods=period).copy()
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = self.calculate_returns(data)
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def detect_gaps(self, data: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """
        Detect price gaps in the data
        
        Args:
            data: OHLCV DataFrame
            threshold: Minimum gap size as percentage
            
        Returns:
            DataFrame with gap information
        """
        gaps = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Gap up: current low > previous high
            if current['Low'] > previous['High']:
                gap_size = (current['Low'] - previous['High']) / previous['High']
                if gap_size >= threshold:
                    gaps.append({
                        'date': current.name,
                        'type': 'Gap Up',
                        'size': gap_size,
                        'previous_high': previous['High'],
                        'current_low': current['Low']
                    })
            
            # Gap down: current high < previous low
            elif current['High'] < previous['Low']:
                gap_size = (previous['Low'] - current['High']) / previous['Low']
                if gap_size >= threshold:
                    gaps.append({
                        'date': current.name,
                        'type': 'Gap Down',
                        'size': gap_size,
                        'previous_low': previous['Low'],
                        'current_high': current['High']
                    })
        
        return pd.DataFrame(gaps)
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """
        Calculate basic support and resistance levels
        
        Args:
            data: OHLCV DataFrame
            window: Window for local min/max calculation
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Calculate local minima and maxima
            lows = data['Low'].rolling(window=window, center=True).min()
            highs = data['High'].rolling(window=window, center=True).max()
            
            # Find support levels (local lows)
            support_levels = []
            for i in range(window, len(data) - window):
                if data['Low'].iloc[i] == lows.iloc[i]:
                    support_levels.append(data['Low'].iloc[i])
            
            # Find resistance levels (local highs)
            resistance_levels = []
            for i in range(window, len(data) - window):
                if data['High'].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(data['High'].iloc[i])
            
            return {
                'nearest_support': min(support_levels) if support_levels else data['Low'].min(),
                'nearest_resistance': max(resistance_levels) if resistance_levels else data['High'].max(),
                'support_levels': sorted(list(set(support_levels)))[-5:],  # Last 5 unique levels
                'resistance_levels': sorted(list(set(resistance_levels)), reverse=True)[:5]  # Top 5 unique levels
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return {
                'nearest_support': data['Low'].min(),
                'nearest_resistance': data['High'].max(),
                'support_levels': [],
                'resistance_levels': []
            }
    
    def get_time_until_next_candle(self, timeframe: str) -> str:
        """Calculate time until next candle formation"""
        now = datetime.now()
        
        timeframe_minutes = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '1hour': 60,
            'daily': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        
        if timeframe == 'daily':
            # Next day at market open
            next_candle = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now >= next_candle:
                next_candle += timedelta(days=1)
        else:
            # Calculate next interval
            current_minute = now.minute
            next_interval = ((current_minute // minutes) + 1) * minutes
            
            if next_interval >= 60:
                next_candle = now.replace(hour=now.hour + 1, minute=next_interval - 60, second=0, microsecond=0)
            else:
                next_candle = now.replace(minute=next_interval, second=0, microsecond=0)
        
        time_diff = next_candle - now
        
        if time_diff.total_seconds() < 60:
            return f"{int(time_diff.total_seconds())} seconds"
        elif time_diff.total_seconds() < 3600:
            return f"{int(time_diff.total_seconds() // 60)} minutes"
        else:
            return f"{int(time_diff.total_seconds() // 3600)} hours"
    
    def export_data_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """Export DataFrame to CSV file"""
        try:
            data.to_csv(filename)
            return True
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return False
