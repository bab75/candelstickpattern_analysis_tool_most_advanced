import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles fetching financial data from various sources"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache for real-time data
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_data(_self, symbol: str, interval: str, period: str = None, start_date=None, end_date=None) -> pd.DataFrame | None:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            start_date: Start date for data (optional, for date range)
            end_date: End date for data (optional, for date range)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert interval format for yfinance
            yf_interval = _self._convert_interval(interval)
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Choose method based on parameters provided
            if period:
                # Use period-based fetching
                data = ticker.history(
                    period=period,
                    interval=yf_interval,
                    prepost=True if period == "1d" else False,
                    auto_adjust=True,
                    back_adjust=False
                )
            elif start_date and end_date:
                # For single day analysis (real-time mode), use special handling
                if start_date == end_date:
                    # Single trading day with extended hours
                    data = ticker.history(
                        period="1d",
                        interval=yf_interval,
                        prepost=True,  # Include pre-market and after-hours
                        auto_adjust=True,
                        back_adjust=False
                    )
                    
                    # If no data for today (weekend/holiday), get last trading day
                    if data.empty:
                        data = ticker.history(
                            period="5d",
                            interval=yf_interval,
                            prepost=True,
                            auto_adjust=True,
                            back_adjust=False
                        )
                        # Keep only the last trading day
                        if not data.empty:
                            last_date = data.index[-1].date()
                            data = data[data.index.date == last_date]
                else:
                    # Historical mode - use date range
                    data = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=yf_interval,
                        prepost=False,
                        auto_adjust=True,
                        back_adjust=False
                    )
            else:
                # Default to 1 month period
                data = ticker.history(
                    period="1mo",
                    interval=yf_interval,
                    auto_adjust=True,
                    back_adjust=False
                )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Clean and prepare data
            data = _self._clean_data(data)
            
            # Add technical indicators
            data = _self._add_technical_indicators(data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            st.error(f"Failed to fetch data for {symbol}: {str(e)}")
            return None
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to yfinance format"""
        interval_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '1hour': '1h',
            'daily': '1d'
        }
        return interval_map.get(interval, '1h')
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the fetched data"""
        # Remove any rows with NaN values
        data = data.dropna().copy()
        
        # Ensure all OHLC values are positive
        data = data[(data['Open'] > 0) & (data['High'] > 0) & 
                   (data['Low'] > 0) & (data['Close'] > 0)].copy()
        
        # Validate OHLC relationships
        data = data[(data['High'] >= data['Open']) & (data['High'] >= data['Close']) &
                   (data['Low'] <= data['Open']) & (data['Low'] <= data['Close'])].copy()
        
        # Sort by timestamp
        data = data.sort_index()
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data"""
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Price change
        data['Price_Change'] = data['Close'].pct_change()
        
        # High-Low spread
        data['HL_Spread'] = data['High'] - data['Low']
        
        # Body size (absolute difference between open and close)
        data['Body_Size'] = abs(data['Close'] - data['Open'])
        
        # Upper and lower shadows
        data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        
        # Candle type
        data['Candle_Type'] = np.where(data['Close'] > data['Open'], 'Bullish', 'Bearish')
        
        return data
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_info(_self, symbol: str) -> dict:
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {'name': symbol, 'sector': 'N/A', 'industry': 'N/A', 'market_cap': 0, 'currency': 'USD'}
    
    def get_real_time_price(self, symbol: str) -> dict | None:
        """Get real-time price information"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                return None
            
            latest = data.iloc[-1]
            return {
                'price': latest['Close'],
                'change': latest['Close'] - data.iloc[-2]['Close'] if len(data) > 1 else 0,
                'volume': latest['Volume'],
                'timestamp': latest.name
            }
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {str(e)}")
            return None
