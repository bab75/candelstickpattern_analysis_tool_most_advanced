# Overview

This is a professional-grade candlestick pattern analysis platform built with Python and Streamlit, designed for serious traders seeking to maximize profit potential. The application provides comprehensive market analysis across four specialized modes: Single Stock Analysis with advanced pattern recognition, Market Scanner for multi-symbol opportunity identification, Advanced Analytics with technical indicators and divergence detection, and Risk Management for position sizing and portfolio protection. 

**Key Enhancement (August 2025):** Real-time mode now focuses on single trading day analysis with extended hours (pre-market 4:00 AM to after-hours 8:00 PM) for comprehensive day trading insights, while historical mode provides traditional date range analysis. Features include AI-powered trading decision engine with 100% confidence analysis, comprehensive external research links to major financial platforms, advanced technical indicators suite, professional trading strategies, and complete project download capability.

**Latest Fixes (August 2025):** FINAL COMPLETE VERSION - All critical issues resolved, comprehensive help documentation added, and two market-demanded enhancement modules integrated. Fixed Portfolio Tracker with session state persistence, eliminated Trading Intelligence issues, added News Sentiment Analyzer and Earnings Calendar modules. Platform is production-ready with professional documentation and stable operation across all modules.

**Major Enhancement (August 2025):** Added advanced Trading Intelligence module with AI-powered market analysis, comprehensive confidence scoring, and professional-grade trading signals. Fixed options analysis to work for all dropdown selections including Strategy Analysis and IV Analysis. Created robust trading platform with market intelligence dashboard, risk-reward analysis, and position sizing recommendations.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Multi-mode Streamlit application with professional navigation and wide layout
- **Visualization**: Advanced Plotly integration with pattern annotations, technical indicators, and volume analysis
- **User Interface**: Mode-based navigation with specialized interfaces for different trading functions
- **Export System**: Comprehensive Excel and CSV export with formatted reports and multiple data sheets
- **Caching**: Optimized Streamlit caching system with proper hash handling for real-time performance

## Backend Architecture
- **Advanced Modular Design**: Professional-grade separation with specialized trading modules:
  - `DataFetcher`: Real-time market data with caching and multi-timeframe support
  - `PatternRecognition`: Advanced algorithmic pattern detection with confidence scoring
  - `ChartGenerator`: Professional interactive visualizations with pattern highlighting
  - `PatternExplanations`: Comprehensive educational content and trading strategies
  - `AdvancedAnalytics`: Technical indicators (RSI, MACD, Bollinger Bands, Stochastic), divergence detection, and signal generation
  - `MarketScanner`: Multi-symbol scanning with concurrent processing and breakout detection
  - `ExportHandler`: Professional reporting with Excel formatting and multiple data sheets
  - `Utils`: Enhanced market utilities with timezone handling and risk calculations
- **Risk Management**: Integrated position sizing, stop-loss calculation, and portfolio risk analysis
- **Backtesting Engine**: Pattern performance analysis with win-rate and return calculations

## Data Architecture
- **Real-time Data**: Yahoo Finance API integration with automatic refresh during market hours
- **Data Models**: OHLCV (Open, High, Low, Close, Volume) DataFrame structure
- **Timeframe Support**: Multiple intervals from 1-minute to daily charts
- **Pattern Detection**: Algorithm-based pattern recognition with confidence scoring and volume confirmation

## Professional Trading Features
- **Advanced Pattern Recognition**: 13+ patterns with confidence scoring, volume confirmation, and trend context analysis
- **Comprehensive Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, Williams %R, CCI, Parabolic SAR, OBV, VWAP, Ichimoku Cloud, Fibonacci retracements, and Pivot Points
- **Professional Trading Strategies**: Mean Reversion, Momentum Breakout, Trend Following, Swing Trading, Scalping, Gap Trading, Volume Spike, and Support/Resistance strategies
- **AI Trading Decision Engine**: Multi-factor analysis with confidence scoring, risk assessment, and actionable trading plans
- **External Research Integration**: Direct links to Yahoo Finance, Google Finance, MarketWatch, Seeking Alpha, NASDAQ, TradingView, Finviz, SEC Filings, StockTwits, and Benzinga
- **Market Intelligence**: Multi-symbol scanning, sector analysis, breakout detection, and market mover identification
- **Advanced Analytics Interface**: Multi-tab dashboard with Core, Advanced, Momentum, and Volume indicators
- **Trading Strategy Consensus**: Automated strategy evaluation with confidence scoring and consensus signals
- **Market Sentiment Analysis**: Comprehensive sentiment scoring based on multiple technical indicators
- **Volume Profile Analysis**: Point of Control (POC) identification and volume distribution analysis
- **Confidence-Based Decision Making**: User-configurable confidence thresholds with clear go/no-go recommendations
- **Risk Management Tools**: Position sizing based on ATR, stop-loss/take-profit calculation, portfolio risk analysis
- **Performance Analytics**: Pattern backtesting, win-rate analysis, risk/reward calculations, and volatility metrics
- **Professional Export**: Formatted Excel reports with multiple sheets, summary statistics, and pattern frequency analysis
- **Real-time Monitoring**: Live market status, session indicators, and automatic data refresh during trading hours

# External Dependencies

## Data Sources
- **Yahoo Finance (yfinance)**: Primary data provider for real-time and historical market data
- **Market Data**: OHLCV data, volume information, and basic company metadata

## Python Libraries
- **Streamlit**: Web application framework and UI components
- **Plotly**: Interactive charting and visualization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and mathematical operations
- **PyTZ**: Timezone handling for market hours calculation

## Deployment Platform
- **Streamlit Cloud**: Recommended deployment platform for easy hosting and sharing
- **Environment**: Python runtime with package dependencies managed via requirements

## APIs and Services
- **Yahoo Finance API**: Free market data access with rate limiting considerations
- **Timezone Services**: PyTZ for accurate market hours across different exchanges
- **Logging**: Python's built-in logging framework for application monitoring