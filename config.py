"""Configuration settings for the Candlestick Pattern Analysis Tool"""

import os
from typing import List

class Config:
    """Application configuration class"""
    
    # Application settings
    APP_NAME = "Candlestick Pattern Analysis Tool"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Data settings
    DEFAULT_SYMBOL = "AAPL"
    DEFAULT_PERIOD = "1mo"  # 1 month
    MAX_DATA_POINTS = 10000
    
    # Supported timeframes
    TIMEFRAMES = [
        "1min",
        "5min", 
        "15min",
        "1hour",
        "daily"
    ]
    
    # Pattern types for analysis
    PATTERN_TYPES = [
        "Hammer",
        "Inverted Hammer",
        "Shooting Star",
        "Hanging Man",
        "Doji",
        "Bullish Engulfing",
        "Bearish Engulfing",
        "Morning Star",
        "Evening Star",
        "Three White Soldiers",
        "Three Black Crows",
        "Piercing Pattern",
        "Dark Cloud Cover"
    ]
    
    # Pattern recognition settings
    MIN_PATTERN_CONFIDENCE = 0.6
    VOLUME_THRESHOLD_MULTIPLIER = 1.5  # Volume must be 1.5x average for confirmation
    
    # Chart settings
    CHART_HEIGHT = 800
    CHART_WIDTH = 1200
    
    # Colors for different elements
    COLORS = {
        'bullish_candle': '#26a69a',
        'bearish_candle': '#ef5350',
        'volume_bars': '#1f77b4',
        'sma_20': '#ff7f0e',
        'sma_50': '#2ca02c',
        'pattern_bullish': '#4caf50',
        'pattern_bearish': '#f44336',
        'pattern_neutral': '#ff9800',
        'background': '#ffffff',
        'grid': '#f0f0f0'
    }
    
    # API settings
    YFINANCE_TIMEOUT = 30  # seconds
    CACHE_DURATION = 300   # 5 minutes for real-time data
    
    # Risk management defaults
    DEFAULT_STOP_LOSS_PERCENT = 2.0
    DEFAULT_TAKE_PROFIT_PERCENT = 4.0
    
    # Educational content settings
    SHOW_PATTERN_EXPLANATIONS = True
    SHOW_TRADING_TIPS = True
    DETAILED_CANDLE_ANALYSIS = True
    
    # Export settings
    EXPORT_FORMATS = ["CSV", "Excel", "JSON"]
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Market hours (Eastern Time)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Technical indicator settings
    SMA_SHORT_PERIOD = 20
    SMA_LONG_PERIOD = 50
    VOLUME_MA_PERIOD = 20
    VOLATILITY_PERIOD = 20
    
    # Pattern filtering
    ENABLE_VOLUME_FILTER = True
    ENABLE_TREND_CONTEXT = True
    MINIMUM_CANDLE_BODY_SIZE = 0.1  # Minimum body size as percentage of total range
    
    # Real-time update settings
    REAL_TIME_UPDATE_INTERVAL = 60  # seconds
    ENABLE_AUTO_REFRESH = True
    
    # GitHub deployment settings
    GITHUB_REPO_URL = "https://github.com/yourusername/candlestick-analysis-tool"
    STREAMLIT_CLOUD_URL = "https://your-app.streamlit.app"
    
    # Performance settings
    ENABLE_CACHING = True
    MAX_CACHE_SIZE = 100  # Maximum number of cached datasets
    
    @classmethod
    def get_timeframe_display_name(cls, timeframe: str) -> str:
        """Get user-friendly display name for timeframe"""
        display_names = {
            "1min": "1 Minute",
            "5min": "5 Minutes",
            "15min": "15 Minutes", 
            "1hour": "1 Hour",
            "daily": "Daily"
        }
        return display_names.get(timeframe, timeframe)
    
    @classmethod
    def get_pattern_category(cls, pattern_type: str) -> str:
        """Get category for a pattern type"""
        categories = {
            "Hammer": "Bullish Reversal",
            "Inverted Hammer": "Bullish Reversal",
            "Bullish Engulfing": "Bullish Reversal",
            "Morning Star": "Bullish Reversal",
            "Piercing Pattern": "Bullish Reversal",
            "Shooting Star": "Bearish Reversal",
            "Hanging Man": "Bearish Reversal",
            "Bearish Engulfing": "Bearish Reversal", 
            "Evening Star": "Bearish Reversal",
            "Dark Cloud Cover": "Bearish Reversal",
            "Three White Soldiers": "Bullish Continuation",
            "Three Black Crows": "Bearish Continuation",
            "Doji": "Indecision"
        }
        return categories.get(pattern_type, "Unknown")
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return os.getenv("STREAMLIT_SHARING_MODE") == "true"
    
    @classmethod
    def get_data_source_info(cls) -> dict:
        """Get information about data sources"""
        return {
            "primary": "Yahoo Finance (yfinance)",
            "real_time_delay": "15-20 minutes",
            "historical_coverage": "Up to 10+ years",
            "supported_exchanges": ["NYSE", "NASDAQ", "LSE", "TSE", "ASX"],
            "data_frequency": "1min to daily",
            "reliability": "High for major stocks"
        }
