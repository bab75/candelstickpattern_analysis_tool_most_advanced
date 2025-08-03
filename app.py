import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from modules.data_fetcher import DataFetcher
from modules.pattern_recognition import PatternRecognition
from modules.chart_generator import ChartGenerator
from modules.pattern_explanations import PatternExplanations
from modules.utils import Utils
from modules.export_handler import ExportHandler
from modules.advanced_analytics import AdvancedAnalytics
from modules.market_scanner import MarketScanner
from modules.decision_engine import DecisionEngine
from modules.advanced_backtesting import AdvancedBacktesting
from modules.portfolio_tracker import PortfolioTracker
from modules.options_analyzer import OptionsAnalyzer
from modules.alert_system import AlertSystem
from modules.trading_strategies import TradingStrategies
from modules.advanced_decision_engine import AdvancedDecisionEngine
from modules.trading_intelligence import TradingIntelligence
from modules.candlestick_profit_analyzer import CandlestickProfitAnalyzer
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Candlestick Pattern Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    st.title("📈 Professional Candlestick Pattern Analysis Tool")
    st.markdown("*Advanced trading analysis with pattern recognition, market scanning, and risk management*")
    
    # Initialize session state for alert system persistence
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = AlertSystem()
    
    # Initialize modules
    data_fetcher = DataFetcher()
    pattern_recognition = PatternRecognition()
    chart_generator = ChartGenerator()
    pattern_explanations = PatternExplanations()
    utils = Utils()
    export_handler = ExportHandler()
    advanced_analytics = AdvancedAnalytics()
    market_scanner = MarketScanner()
    decision_engine = DecisionEngine()
    advanced_backtesting = AdvancedBacktesting()
    # Initialize portfolio tracker with session state persistence
    if 'portfolio_tracker' not in st.session_state:
        st.session_state.portfolio_tracker = PortfolioTracker()
    portfolio_tracker = st.session_state.portfolio_tracker
    options_analyzer = OptionsAnalyzer()
    alert_system = st.session_state.alert_system  # Use persistent alert system
    trading_strategies = TradingStrategies()
    advanced_decision_engine = AdvancedDecisionEngine()
    trading_intelligence = TradingIntelligence()
    profit_analyzer = CandlestickProfitAnalyzer()
    
    # Main navigation
    st.sidebar.title("🎯 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Stock Analysis", "Profit Confirmation Analysis", "Market Scanner", "Advanced Analytics", "AI Trading Decision", "Trading Intelligence",
         "Risk Management", "Trading Advisor", "Advanced Backtesting", "Portfolio Tracker", "Options Analyzer", "Alert System", "News Sentiment", "Earnings Calendar"],
        help="Select the type of analysis you want to perform"
    )
    
    # Route to different app modes
    if app_mode == "Single Stock Analysis":
        single_stock_analysis(data_fetcher, pattern_recognition, chart_generator, 
                             pattern_explanations, utils, export_handler, advanced_analytics)
    

    
    elif app_mode == "Profit Confirmation Analysis":
        profit_confirmation_interface(data_fetcher, pattern_recognition, profit_analyzer)
    
    elif app_mode == "Market Scanner":
        market_scanner_interface(market_scanner, pattern_recognition)
    
    elif app_mode == "Advanced Analytics":
        advanced_analytics_interface(data_fetcher, advanced_analytics, pattern_recognition, trading_strategies)
    
    elif app_mode == "AI Trading Decision":
        ai_trading_decision_interface(data_fetcher, advanced_analytics, pattern_recognition, 
                                    trading_strategies, advanced_decision_engine)
    
    elif app_mode == "Trading Intelligence":
        trading_intelligence_interface(data_fetcher, advanced_analytics, pattern_recognition, 
                                     trading_intelligence, alert_system)
    
    elif app_mode == "Risk Management":
        risk_management_interface(data_fetcher, advanced_analytics)
    
    elif app_mode == "Trading Advisor":
        trading_advisor_interface(data_fetcher, pattern_recognition, advanced_analytics, decision_engine)
    
    elif app_mode == "Advanced Backtesting":
        backtesting_interface(advanced_backtesting, pattern_recognition)
    
    elif app_mode == "Portfolio Tracker":
        portfolio_interface(portfolio_tracker, data_fetcher, advanced_analytics)
    
    elif app_mode == "Options Analyzer":
        options_interface(options_analyzer, data_fetcher)
    
    elif app_mode == "Alert System":
        alert_interface(alert_system, data_fetcher, pattern_recognition, advanced_analytics)
    
    elif app_mode == "News Sentiment":
        from modules.news_sentiment_analyzer import news_sentiment_interface
        news_sentiment_interface()
    
    elif app_mode == "Earnings Calendar":
        from modules.earnings_calendar import earnings_calendar_interface
        earnings_calendar_interface()

def single_stock_analysis(data_fetcher, pattern_recognition, chart_generator, 
                         pattern_explanations, utils, export_handler, advanced_analytics):
    """Single stock analysis interface"""
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stock symbol input
        symbol = st.text_input(
            "Stock Symbol", 
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        # Analysis mode
        analysis_mode = st.radio(
            "Analysis Mode",
            options=["Real-time", "Historical"],
            help="Choose between real-time analysis or historical date range"
        )
        
        # Timeframe selection with mode-specific defaults
        if analysis_mode == "Real-time":
            default_idx = 2  # Default to 15min for real-time
            help_text = "Real-time mode: Single trading day with extended hours (pre-market 4AM + after-hours 8PM). Best timeframes: 1min-1hour"
        else:
            default_idx = 4  # Default to daily for historical
            help_text = "Historical analysis: Recommended timeframes are daily or weekly for pattern reliability"
        
        timeframe = st.selectbox(
            "Timeframe",
            options=Config.TIMEFRAMES,
            index=default_idx,
            help=help_text
        )
        
        if analysis_mode == "Historical":
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30),
                    max_value=datetime.now().date()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    max_value=datetime.now().date()
                )
        else:
            # Real-time mode focuses on single trading day with extended hours
            end_date = datetime.now().date()
            start_date = end_date  # Same day for intraday analysis
            
            st.info("🕐 **Real-time Mode**: Analyzing current trading day with pre-market (4:00 AM) and after-hours (8:00 PM) data for comprehensive day trading insights.")
        
        # Pattern filters
        st.subheader("🔍 Pattern Filters")
        pattern_types = st.multiselect(
            "Select Pattern Types",
            options=Config.PATTERN_TYPES,
            default=Config.PATTERN_TYPES,
            help="Choose which pattern types to analyze"
        )
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze Patterns", type="primary", use_container_width=True)
        
        # Export options
        st.subheader("📤 Export Options")
        export_format = st.selectbox(
            "Export Format",
            options=["Excel", "CSV"],
            help="Choose format for exporting pattern analysis results"
        )
    
    # Main content area
    if analyze_button:
        if not symbol:
            st.error("Please enter a valid stock symbol")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Fetch data
            status_text.text("📊 Fetching market data...")
            progress_bar.progress(20)
            
            if analysis_mode == "Real-time":
                data = data_fetcher.fetch_data(symbol, timeframe, period="1d")
            else:
                data = data_fetcher.fetch_data(symbol, timeframe, start_date=start_date, end_date=end_date)
            
            if data is None or data.empty:
                st.error(f"No data available for {symbol} in the selected date range")
                return
            
            # Validate data is DataFrame
            if not isinstance(data, pd.DataFrame):
                logger.error(f"Data fetcher returned {type(data)} instead of DataFrame")
                st.error("Invalid data format received")
                return
            
            progress_bar.progress(40)
            
            # Analyze patterns
            status_text.text("🔍 Analyzing candlestick patterns...")
            patterns = pattern_recognition.analyze_patterns(data, pattern_types)
            progress_bar.progress(60)
            
            # Generate charts
            status_text.text("📈 Generating interactive charts...")
            fig = chart_generator.create_candlestick_chart(data, patterns, symbol, timeframe)
            progress_bar.progress(80)
            
            # Advanced analytics
            status_text.text("🧠 Generating advanced analytics...")
            try:
                logger.info(f"Data type before trading signals: {type(data)}")
                trading_signals = advanced_analytics.generate_trading_signals(data, patterns)
                logger.info(f"Trading signals generated successfully")
                
                risk_metrics = advanced_analytics.calculate_risk_metrics(data)
                logger.info(f"Risk metrics calculated successfully")
                
                backtest_results = advanced_analytics.backtest_patterns(data, patterns)
                logger.info(f"Backtest completed successfully")
            except Exception as e:
                logger.error(f"Error in advanced analytics: {str(e)}")
                trading_signals = []
                risk_metrics = {'stop_loss_level': 0, 'take_profit_level': 0, 'position_size_shares': 0}
                backtest_results = {'total_trades': 0, 'win_rate': 0, 'avg_return': 0}
            
            # Clear progress indicators
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 {symbol} Candlestick Chart ({timeframe})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Real-time info for current mode
                if analysis_mode == "Real-time":
                    st.info("🔄 Real-time mode: Data updates automatically during market hours")
            
            with col2:
                st.subheader("📋 Pattern Analysis Summary")
                
                if patterns:
                    # Pattern statistics
                    pattern_counts = {}
                    for pattern in patterns:
                        pattern_type = pattern['type']
                        if pattern_type in pattern_counts:
                            pattern_counts[pattern_type] += 1
                        else:
                            pattern_counts[pattern_type] = 1
                    
                    st.markdown("**Pattern Frequency:**")
                    for pattern_type, count in pattern_counts.items():
                        st.write(f"• {pattern_type}: {count}")
                    
                    # Latest market info
                    latest_data = data.iloc[-1]
                    st.markdown("**Latest Market Data:**")
                    st.write(f"• Close: ${latest_data['Close']:.2f}")
                    st.write(f"• Volume: {latest_data['Volume']:,}")
                    
                    # Market session indicator
                    market_status = utils.get_market_status(symbol)
                    if market_status['is_open']:
                        st.success(f"🟢 Market Open - {market_status['session']}")
                    else:
                        st.info(f"🔴 Market Closed - Next: {market_status['next_open']}")
                    
                    # Trading signals
                    if trading_signals and isinstance(trading_signals, dict) and 'signals' in trading_signals:
                        st.markdown("**🎯 Trading Signals:**")
                        for signal in trading_signals['signals']:
                            if isinstance(signal, dict) and 'signal' in signal and 'type' in signal:
                                color = "green" if "Buy" in str(signal['signal']) else "red" if "Sell" in str(signal['signal']) else "orange"
                                st.markdown(f"<div style='color: {color}'>{signal['type']}: {signal['signal']}</div>", 
                                          unsafe_allow_html=True)
                    elif trading_signals and isinstance(trading_signals, list):
                        st.markdown("**🎯 Trading Signals:**")
                        for signal in trading_signals:
                            if isinstance(signal, dict) and 'signal' in signal and 'type' in signal:
                                color = "green" if "Buy" in str(signal['signal']) else "red" if "Sell" in str(signal['signal']) else "orange"
                                st.markdown(f"<div style='color: {color}'>{signal['type']}: {signal['signal']}</div>", 
                                          unsafe_allow_html=True)
                    
                    # Risk metrics
                    st.markdown("**⚠️ Risk Management:**")
                    st.write(f"• Stop Loss: ${risk_metrics['stop_loss_level']:.2f}")
                    st.write(f"• Take Profit: ${risk_metrics['take_profit_level']:.2f}")
                    st.write(f"• Position Size: {risk_metrics['position_size_shares']} shares")
                    
                    # Backtest performance
                    if backtest_results['total_trades'] > 0:
                        st.markdown("**📊 Pattern Performance:**")
                        st.write(f"• Win Rate: {backtest_results['win_rate']:.1%}")
                        st.write(f"• Avg Return: {backtest_results['avg_return']:.1%}")
                        st.write(f"• Total Trades: {backtest_results['total_trades']}")
                    
                    # External research links
                    st.markdown("**🔗 External Research Links:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"📈 [Yahoo Finance](https://finance.yahoo.com/quote/{symbol})")
                        st.markdown(f"📊 [MarketWatch](https://www.marketwatch.com/investing/stock/{symbol.lower()})")
                        st.markdown(f"💹 [TradingView Chart](https://www.tradingview.com/symbols/{symbol}/)")
                        st.markdown(f"📱 [StockTwits](https://stocktwits.com/symbol/{symbol})")
                        st.markdown(f"📈 [StockChartsAI](https://stockchartsai.com/chart_search.php?symbol={symbol})")
                    
                    with col2:
                        st.markdown(f"📰 [Seeking Alpha](https://seekingalpha.com/symbol/{symbol})")
                        st.markdown(f"🏢 [SEC Filings](https://www.sec.gov/edgar/search/#/q={symbol})")
                        st.markdown(f"📋 [Finviz](https://finviz.com/quote.ashx?t={symbol.lower()})")
                        st.markdown(f"📊 [NASDAQ](https://www.nasdaq.com/market-activity/stocks/{symbol.lower()}/)")
                    
                    # Navigation to dedicated features
                    st.markdown("**⚡ Quick Access:**")
                    nav_col1, nav_col2 = st.columns(2)
                    
                    with nav_col1:
                        st.info("📱 **Alert System**: Complete alert management in dedicated tab")
                    
                    with nav_col2:
                        st.info("💰 **Profit Analysis**: Go to Profit Confirmation tab for detailed profit analysis")
                else:
                    st.info("No patterns detected in the selected timeframe and criteria")
                
                # Reference to dedicated Profit Analysis tab
                if patterns:
                    st.markdown("---")
                    st.info("💰 **For detailed Profit Confirmation Analysis**, visit the 'Profit Confirmation' tab in the navigation menu. It provides comprehensive profit analysis with success rates, risk-reward calculations, and trading strategies.")
            
            # Export section
            if patterns:
                with st.sidebar:
                    st.markdown("---")
                    st.subheader("📥 Export Results")
                    export_handler.create_download_button(
                        patterns, data, symbol, timeframe, export_format
                    )
            
            # Detailed pattern explanations
            if patterns:
                st.subheader("📚 Pattern Explanations & Trading Insights")
                
                for i, pattern in enumerate(patterns):
                    try:
                        # Validate pattern structure
                        if not isinstance(pattern, dict) or 'type' not in pattern:
                            logger.error(f"Invalid pattern structure: {pattern}")
                            continue
                            
                        with st.expander(f"{pattern['type']} - {pattern.get('date', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.1%})"):
                            explanation = pattern_explanations.get_explanation(pattern['type'])
                            
                            # Validate explanation structure
                            if not isinstance(explanation, dict):
                                logger.error(f"Invalid explanation type: {type(explanation)} for pattern {pattern['type']}")
                                st.error(f"Unable to load explanation for {pattern['type']}")
                                continue
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**Pattern Description:**")
                                st.write(explanation.get('description', 'Description not available'))
                                
                                st.markdown("**Formation Process:**")
                                st.write(explanation.get('formation', 'Formation details not available'))
                            
                            with col2:
                                st.markdown("**Trading Significance:**")
                                st.write(explanation.get('significance', 'Significance not available'))
                                
                                st.markdown("**Trading Strategy:**")
                                st.write(explanation.get('strategy', 'Strategy not available'))
                        
                            # Pattern-specific data (inside expander)
                            try:
                                if hasattr(data, 'iloc') and isinstance(data, pd.DataFrame):
                                    pattern_data = data.iloc[pattern['start_idx']:pattern['end_idx']+1]
                                    st.markdown("**Candlestick Formation Details:**")
                                    
                                    for idx, row in pattern_data.iterrows():
                                        body_size = abs(row['Close'] - row['Open'])
                                        upper_shadow = row['High'] - max(row['Open'], row['Close'])
                                        lower_shadow = min(row['Open'], row['Close']) - row['Low']
                                        candle_type = "Bullish" if row['Close'] > row['Open'] else "Bearish"
                                        
                                        st.write(f"**{idx.strftime('%Y-%m-%d %H:%M')}** - {candle_type}")
                                        st.write(f"  • Open: ${row['Open']:.2f}, Close: ${row['Close']:.2f}")
                                        st.write(f"  • High: ${row['High']:.2f}, Low: ${row['Low']:.2f}")
                                        st.write(f"  • Body: ${body_size:.2f}, Upper Shadow: ${upper_shadow:.2f}, Lower Shadow: ${lower_shadow:.2f}")
                                else:
                                    st.warning("Unable to display candlestick formation details - invalid data format")
                                    logger.error(f"Data type error: expected DataFrame, got {type(data)}")
                            except Exception as e:
                                st.warning("Unable to display candlestick formation details")
                                logger.error(f"Error displaying pattern data: {str(e)}, data type: {type(data)}")
                    except Exception as e:
                        logger.error(f"Error processing pattern {i}: {str(e)}")
                        st.error(f"Error displaying pattern: {str(e)}")
                        continue
            
            # Educational section
            with st.expander("📖 Learn About Candlestick Analysis"):
                st.markdown("""
                ### Understanding Candlestick Patterns
                
                Candlestick patterns are formed by the price action of securities and provide insights into market sentiment and potential price movements.
                
                **Key Components of a Candlestick:**
                - **Body**: The area between opening and closing prices
                - **Wicks/Shadows**: Lines extending from the body to the high and low prices
                - **Color**: Green/White for bullish (close > open), Red/Black for bearish (close < open)
                
                **Pattern Categories:**
                - **Reversal Patterns**: Signal potential trend changes
                - **Continuation Patterns**: Suggest trend continuation
                - **Indecision Patterns**: Show market uncertainty
                
                **Trading Timeframes:**
                - **1min-15min**: Scalping and day trading
                - **1hour-4hour**: Intraday swing trading
                - **Daily**: Position and swing trading
                """)
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error in analysis: {str(e)}")
            logger.error(f"Full traceback: {error_traceback}")
            
            # Log variable types for debugging
            logger.error(f"Variable types at error: patterns={type(patterns) if 'patterns' in locals() else 'undefined'}, data={type(data) if 'data' in locals() else 'undefined'}, trading_signals={type(trading_signals) if 'trading_signals' in locals() else 'undefined'}")
            if 'patterns' in locals() and patterns:
                logger.error(f"First pattern type: {type(patterns[0])}")
            if 'trading_signals' in locals() and trading_signals:
                logger.error(f"Trading signals structure: {trading_signals if isinstance(trading_signals, (str, int, float)) else type(trading_signals)}")
                
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please check your internet connection and try again with a valid stock symbol.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Candlestick Pattern Analysis Tool
        
        This tool provides comprehensive analysis of candlestick patterns with the following features:
        
        ### 🚀 Key Features
        - **Real-time Data**: Live market data updates during trading hours
        - **Historical Analysis**: Analyze patterns across custom date ranges
        - **Multiple Timeframes**: Support for 1min, 5min, 15min, 1hour, and daily charts
        - **Pattern Recognition**: Automated detection of major candlestick patterns
        - **Educational Content**: Detailed explanations of pattern formation and trading significance
        - **Interactive Charts**: Plotly-powered interactive candlestick visualizations
        
        ### 📊 Supported Patterns (13+ Core Patterns)
        **Reversal Patterns:** Hammer, Inverted Hammer, Shooting Star, Hanging Man, Doji, 
        Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Pattern, Dark Cloud Cover
        
        **Continuation Patterns:** Three White Soldiers, Three Black Crows
        
        **Additional patterns available for expansion and customization**
        
        ### 🎯 Getting Started
        1. Enter a stock symbol in the sidebar (e.g., AAPL, GOOGL, TSLA)
        2. Choose your preferred timeframe
        3. Select real-time or historical analysis mode
        4. Configure pattern filters
        5. Click "Analyze Patterns" to begin
        
        **Ready to analyze? Configure your settings in the sidebar and click the analyze button!**
        """)
        
        # Download section
        st.markdown("---")
        st.markdown("### 📁 Download Complete Project")
        st.markdown("**Get all source code and deployment files**")
        
        # Check if files exist
        import os
        zip_path = "/home/runner/workspace/trading_platform_complete.zip"
        deployment_req_path = "/home/runner/workspace/deployment_requirements.txt"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(zip_path):
                with open(zip_path, "rb") as zip_file:
                    st.download_button(
                        label="📦 Download Complete Project ZIP",
                        data=zip_file.read(),
                        file_name="trading_platform_complete.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
            else:
                st.error("Project ZIP not available")
        
        with col2:
            if os.path.exists(deployment_req_path):
                with open(deployment_req_path, "r") as req_file:
                    st.download_button(
                        label="📋 Download Requirements.txt",
                        data=req_file.read(),
                        file_name="requirements.txt",
                        mime="text/plain",
                        type="secondary",
                        use_container_width=True
                    )
            else:
                st.error("Requirements file not available")
        
        with st.expander("📋 Deployment Instructions"):
            st.markdown("""
            **🚀 Quick Deployment Guide:**
            
            1. **Download Files:** Get both ZIP and requirements.txt files above
            2. **Extract:** Unzip the complete project package  
            3. **Install Dependencies:** `pip install -r requirements.txt`
            4. **Run Locally:** `streamlit run app.py`
            5. **Cloud Deploy:** Upload to Streamlit Cloud, Heroku, or Replit
            
            **📦 Package Contains:**
            - Complete source code (all Python modules)
            - Professional trading algorithms & pattern recognition
            - Portfolio management & options analysis tools
            - Real-time alert system & advanced backtesting
            - Export functionality & interactive charts
            - Ready-to-deploy configuration files
            - Professional-grade risk management tools
            
            **🔧 Platform Compatibility:**
            - Streamlit Cloud (recommended)
            - Heroku, Railway, Render
            - Replit, CodeSandbox
            - Local development environment
            
            **💡 Customization Notes:**
            - All modules are well-documented
            - Pattern recognition algorithms are modular
            - Easy to add new patterns or modify existing ones
            - Professional-grade codebase suitable for enhancement
            """)

def market_scanner_interface(market_scanner, pattern_recognition):
    """Market scanner interface"""
    st.header("🔍 Market Scanner")
    st.markdown("Scan multiple stocks for candlestick patterns and trading opportunities")
    
    # Scanner options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scan_type = st.selectbox(
            "Scan Type",
            ["Popular Stocks", "Sector Analysis", "Custom Watchlist", "Crypto", "Market Movers"]
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["1d", "1h", "15min"],
            index=0
        )
    
    with col3:
        max_results = st.number_input(
            "Max Results",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Maximum number of symbols to scan (5-100)"
        )
    
    # Custom watchlist input
    if scan_type == "Custom Watchlist":
        watchlist_input = st.text_area(
            "Enter symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL,AMZN,TSLA",
            help="Enter stock symbols separated by commas"
        )
        symbols = [s.strip().upper() for s in watchlist_input.split(',') if s.strip()]
    
    elif scan_type == "Popular Stocks":
        symbols = market_scanner.popular_stocks[:max_results]
    
    elif scan_type == "Crypto":
        symbols = market_scanner.crypto_symbols
    
    elif scan_type == "Sector Analysis":
        sectors = market_scanner.get_sector_analysis()
        selected_sector = st.selectbox("Select Sector", list(sectors.keys()))
        symbols = sectors[selected_sector]
    
    else:  # Market Movers
        symbols = market_scanner.popular_stocks[:max_results]
    
    # Scan button
    if st.button("🚀 Start Market Scan", type="primary"):
        with st.spinner("Scanning market for patterns..."):
            
            if scan_type == "Market Movers":
                st.subheader("📈 Top Market Movers")
                movers = market_scanner.get_market_movers()
                
                if movers:
                    df_movers = pd.DataFrame(movers)
                    st.dataframe(df_movers, use_container_width=True)
                    
                    # Scan movers for patterns
                    mover_symbols = [m['symbol'] for m in movers[:max_results]]
                    results = market_scanner.scan_watchlist(mover_symbols, pattern_recognition, timeframe)
                    market_scanner.create_watchlist_dashboard(results)
                else:
                    st.info("No significant market movers found today.")
            
            else:
                # Regular pattern scanning
                results = market_scanner.scan_watchlist(symbols, pattern_recognition, timeframe)
                market_scanner.create_watchlist_dashboard(results)
                
                # Breakout analysis
                if scan_type in ["Popular Stocks", "Custom Watchlist"]:
                    st.subheader("🎯 Breakout Analysis")
                    breakouts = market_scanner.scan_for_breakouts(symbols[:15])
                    
                    if breakouts:
                        df_breakouts = pd.DataFrame(breakouts)
                        st.dataframe(df_breakouts, use_container_width=True)
                    else:
                        st.info("No significant breakouts detected.")

def advanced_analytics_interface(data_fetcher, advanced_analytics, pattern_recognition, trading_strategies):
    """Enhanced advanced analytics interface with comprehensive indicators and strategies"""
    st.header("🧠 Advanced Analytics & Trading Strategies")
    st.markdown("Professional-grade technical analysis with multiple indicators, trading strategies, and actionable signals")
    
    # Symbol input and configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15min"], index=0)
    with col3:
        analysis_type = st.selectbox("Analysis Type", ["Complete Analysis", "Quick Scan", "Strategy Focus"])
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            lookback_days = st.slider("Lookback Period (Days)", 30, 365, 90)
        with col2:
            min_confidence = st.slider("Min Signal Confidence", 0.1, 0.9, 0.5)
        with col3:
            show_fibonacci = st.checkbox("Show Fibonacci Levels", True)
    
    if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
        with st.spinner("Performing comprehensive analysis..."):
            # Fetch data
            if lookback_days <= 30:
                period = "1mo"
            elif lookback_days <= 90:
                period = "3mo"
            elif lookback_days <= 180:
                period = "6mo"
            else:
                period = "1y"
            
            data = data_fetcher.fetch_data(symbol, timeframe, period=period)
            
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                
                # === MARKET OVERVIEW ===
                st.subheader("📈 Market Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}%")
                
                with col2:
                    volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x", "High" if volume_ratio > 1.5 else "Normal")
                
                with col3:
                    volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                    st.metric("Volatility (20d)", f"{volatility:.1f}%")
                
                with col4:
                    atr = advanced_analytics.calculate_atr(data).iloc[-1]
                    st.metric("ATR", f"${atr:.2f}")
                
                # === TECHNICAL INDICATORS DASHBOARD ===
                st.subheader("🔬 Technical Indicators Dashboard")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Core Indicators", "📈 Advanced Indicators", "🌊 Momentum", "📊 Volume"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # RSI
                        rsi = advanced_analytics.calculate_rsi(data)
                        current_rsi = rsi.iloc[-1]
                        rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
                        st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                        
                        # MACD
                        macd = advanced_analytics.calculate_macd(data)
                        macd_signal = "🟢 Bullish" if macd['macd'].iloc[-1] > macd['signal'].iloc[-1] else "🔴 Bearish"
                        st.metric("MACD", f"{macd['macd'].iloc[-1]:.4f}", macd_signal)
                        
                        # Bollinger Bands
                        bb = advanced_analytics.calculate_bollinger_bands(data)
                        bb_position = "Upper" if current_price > bb['upper'].iloc[-1] else "Lower" if current_price < bb['lower'].iloc[-1] else "Middle"
                        st.metric("Bollinger Position", bb_position)
                    
                    with col2:
                        # Stochastic
                        stoch = advanced_analytics.calculate_stochastic(data)
                        stoch_signal = "🔴 Overbought" if stoch['k'].iloc[-1] > 80 else "🟢 Oversold" if stoch['k'].iloc[-1] < 20 else "🟡 Neutral"
                        st.metric("Stochastic %K", f"{stoch['k'].iloc[-1]:.1f}", stoch_signal)
                        
                        # Williams %R
                        williams_r = advanced_analytics.calculate_williams_r(data)
                        if not williams_r.empty:
                            williams_signal = "🔴 Overbought" if williams_r.iloc[-1] > -20 else "🟢 Oversold" if williams_r.iloc[-1] < -80 else "🟡 Neutral"
                            st.metric("Williams %R", f"{williams_r.iloc[-1]:.1f}", williams_signal)
                        
                        # CCI
                        cci = advanced_analytics.calculate_cci(data)
                        if not cci.empty:
                            cci_signal = "🔴 Overbought" if cci.iloc[-1] > 100 else "🟢 Oversold" if cci.iloc[-1] < -100 else "🟡 Neutral"
                            st.metric("CCI", f"{cci.iloc[-1]:.1f}", cci_signal)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ichimoku Cloud
                        ichimoku = advanced_analytics.calculate_ichimoku(data)
                        if ichimoku:
                            st.write("**Ichimoku Cloud Analysis:**")
                            if not ichimoku['tenkan_sen'].empty:
                                st.write(f"• Tenkan-sen: ${ichimoku['tenkan_sen'].iloc[-1]:.2f}")
                                st.write(f"• Kijun-sen: ${ichimoku['kijun_sen'].iloc[-1]:.2f}")
                                cloud_signal = "🟢 Above Cloud" if current_price > max(ichimoku['senkou_span_a'].iloc[-1], ichimoku['senkou_span_b'].iloc[-1]) else "🔴 Below Cloud"
                                st.write(f"• Cloud Position: {cloud_signal}")
                        
                        # Parabolic SAR
                        psar = advanced_analytics.calculate_parabolic_sar(data)
                        if not psar.empty:
                            psar_signal = "🟢 Bullish" if current_price > psar.iloc[-1] else "🔴 Bearish"
                            st.metric("Parabolic SAR", f"${psar.iloc[-1]:.2f}", psar_signal)
                    
                    with col2:
                        # Fibonacci Levels
                        if show_fibonacci:
                            fibonacci = advanced_analytics.calculate_fibonacci_levels(data)
                            if fibonacci:
                                st.write("**Fibonacci Retracements:**")
                                for level, price in fibonacci.items():
                                    if 'fib_' in level:
                                        distance = abs(current_price - price) / current_price * 100
                                        proximity = "🎯" if distance < 1 else ""
                                        st.write(f"• {level.replace('fib_', '').replace('_', '.')}: ${price:.2f} {proximity}")
                        
                        # Pivot Points
                        pivots = advanced_analytics.calculate_pivot_points(data)
                        if pivots:
                            st.write("**Daily Pivot Points:**")
                            st.write(f"• Pivot: ${pivots['pivot']:.2f}")
                            st.write(f"• R1: ${pivots['r1']:.2f} | S1: ${pivots['s1']:.2f}")
                            st.write(f"• R2: ${pivots['r2']:.2f} | S2: ${pivots['s2']:.2f}")
                
                with tab3:
                    # Volume-based indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # OBV
                        obv = advanced_analytics.calculate_obv(data)
                        if not obv.empty:
                            obv_change = (obv.iloc[-1] - obv.iloc[-2]) / abs(obv.iloc[-2]) * 100 if obv.iloc[-2] != 0 else 0
                            st.metric("OBV Change", f"{obv_change:+.1f}%")
                        
                        # VWAP
                        vwap = advanced_analytics.calculate_vwap(data)
                        if not vwap.empty:
                            vwap_signal = "🟢 Above VWAP" if current_price > vwap.iloc[-1] else "🔴 Below VWAP"
                            st.metric("VWAP", f"${vwap.iloc[-1]:.2f}", vwap_signal)
                    
                    with col2:
                        # Volume Profile
                        volume_profile = advanced_analytics.calculate_volume_profile(data)
                        if volume_profile.get('poc_price'):
                            poc_distance = (current_price - volume_profile['poc_price']) / volume_profile['poc_price'] * 100
                            st.metric("POC Distance", f"{poc_distance:+.1f}%", f"POC: ${volume_profile['poc_price']:.2f}")
                
                with tab4:
                    # Advanced momentum indicators
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rate of Change
                        roc_5 = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6] * 100) if len(data) > 5 else 0
                        roc_10 = ((current_price - data['Close'].iloc[-11]) / data['Close'].iloc[-11] * 100) if len(data) > 10 else 0
                        st.metric("ROC (5 periods)", f"{roc_5:+.2f}%")
                        st.metric("ROC (10 periods)", f"{roc_10:+.2f}%")
                    
                    with col2:
                        # Money Flow Index (approximation)
                        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                        money_flow = typical_price * data['Volume']
                        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
                        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
                        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
                        if not mfi.empty:
                            mfi_signal = "🔴 Overbought" if mfi.iloc[-1] > 80 else "🟢 Oversold" if mfi.iloc[-1] < 20 else "🟡 Neutral"
                            st.metric("MFI", f"{mfi.iloc[-1]:.1f}", mfi_signal)
                
                # === TRADING STRATEGIES ===
                st.subheader("🎯 Professional Trading Strategies")
                
                strategy_results = trading_strategies.run_all_strategies(data)
                
                # Consensus Signal
                col1, col2, col3 = st.columns(3)
                with col1:
                    signal_color = "🟢" if strategy_results['consensus_signal'] == 'BUY' else "🔴" if strategy_results['consensus_signal'] == 'SELL' else "🟡"
                    st.metric("Consensus Signal", f"{signal_color} {strategy_results['consensus_signal']}", 
                             f"Confidence: {strategy_results['consensus_confidence']:.1%}")
                
                with col2:
                    st.metric("Active Strategies", f"{strategy_results['active_strategies']}/{strategy_results['total_strategies']}")
                
                with col3:
                    high_confidence_strategies = len([s for s in strategy_results.get('strategy_results', []) if s.get('confidence', 0) >= min_confidence])
                    st.metric("High Confidence", f"{high_confidence_strategies}")
                
                # Strategy Details
                if strategy_results.get('strategy_results'):
                    st.write("**Strategy Breakdown:**")
                    
                    for i, strategy in enumerate(strategy_results['strategy_results'][:6]):  # Show top 6
                        if strategy.get('confidence', 0) >= min_confidence:
                            with st.expander(f"{strategy['strategy']} - {strategy['signal']} (Confidence: {strategy['confidence']:.1%})"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"**Entry:** ${strategy.get('entry_price', 0):.2f}")
                                    st.write(f"**Stop Loss:** ${strategy.get('stop_loss', 0):.2f}")
                                
                                with col2:
                                    st.write(f"**Take Profit:** ${strategy.get('take_profit', 0):.2f}")
                                    st.write(f"**Risk/Reward:** {strategy.get('risk_reward', 0):.1f}")
                                
                                with col3:
                                    if 'indicators' in strategy:
                                        st.write("**Key Indicators:**")
                                        for key, value in strategy['indicators'].items():
                                            if isinstance(value, (int, float)):
                                                st.write(f"• {key}: {value:.2f}")
                                            else:
                                                st.write(f"• {key}: {value}")
                
                # === MARKET SENTIMENT ===
                st.subheader("📊 Market Sentiment Analysis")
                
                # Calculate sentiment score
                bullish_indicators = 0
                bearish_indicators = 0
                total_indicators = 0
                
                # RSI sentiment
                if current_rsi < 30:
                    bullish_indicators += 1
                elif current_rsi > 70:
                    bearish_indicators += 1
                total_indicators += 1
                
                # MACD sentiment
                if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]:
                    bullish_indicators += 1
                else:
                    bearish_indicators += 1
                total_indicators += 1
                
                # Volume sentiment
                if volume_ratio > 1.5:
                    if price_change > 0:
                        bullish_indicators += 1
                    else:
                        bearish_indicators += 1
                total_indicators += 1
                
                # Trend sentiment (using moving averages)
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) > 50 else ma_20
                
                if current_price > ma_20 > ma_50:
                    bullish_indicators += 2
                elif current_price < ma_20 < ma_50:
                    bearish_indicators += 2
                total_indicators += 2
                
                # Display sentiment
                sentiment_score = (bullish_indicators - bearish_indicators) / total_indicators if total_indicators > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if sentiment_score > 0.3:
                        st.success(f"🟢 Bullish Sentiment ({sentiment_score:.1%})")
                    elif sentiment_score < -0.3:
                        st.error(f"🔴 Bearish Sentiment ({sentiment_score:.1%})")
                    else:
                        st.warning(f"🟡 Neutral Sentiment ({sentiment_score:.1%})")
                
                with col2:
                    st.metric("Bullish Signals", bullish_indicators)
                
                with col3:
                    st.metric("Bearish Signals", bearish_indicators)
            
            else:
                st.error("Could not fetch data for the specified symbol.")

def ai_trading_decision_interface(data_fetcher, advanced_analytics, pattern_recognition, 
                                trading_strategies, advanced_decision_engine):
    """AI-powered comprehensive trading decision interface"""
    st.header("🤖 AI Trading Decision Engine")
    st.markdown("**Professional-grade AI analysis with maximum confidence scoring and external research links**")
    
    # Input section
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
    with col2:
        timeframe = st.selectbox("Analysis Timeframe", ["1d", "1h", "15min"], index=0)
    with col3:
        analysis_depth = st.selectbox("Analysis Depth", ["Complete Analysis", "Quick Decision", "Deep Research"])
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            confidence_threshold = st.slider("Minimum Confidence (%)", 50, 95, 70)
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        with col3:
            investment_horizon = st.selectbox("Investment Horizon", ["Day Trading", "Swing Trading", "Position Trading"])
    
    if st.button("🚀 Generate AI Trading Decision", type="primary", use_container_width=True):
        with st.spinner("🧠 AI analyzing market data and generating comprehensive decision..."):
            # Fetch comprehensive data
            lookback_days = 120 if analysis_depth == "Deep Research" else 90 if analysis_depth == "Complete Analysis" else 30
            
            if lookback_days <= 30:
                period = "1mo"
            elif lookback_days <= 90:
                period = "3mo"
            else:
                period = "6mo"
            
            data = data_fetcher.fetch_data(symbol, timeframe, period=period)
            
            if data is not None and not data.empty:
                # Generate all required analyses
                # Get all available pattern types for analysis
                pattern_types = ['Hammer', 'Inverted Hammer', 'Shooting Star', 'Hanging Man', 'Doji', 
                               'Bullish Engulfing', 'Bearish Engulfing', 'Morning Star', 'Evening Star',
                               'Three White Soldiers', 'Three Black Crows', 'Piercing Pattern', 'Dark Cloud Cover']
                patterns = pattern_recognition.analyze_patterns(data, pattern_types)
                technical_signals = advanced_analytics.generate_trading_signals(data, patterns)
                strategy_results = trading_strategies.run_all_strategies(data)
                
                # Create market sentiment data
                current_price = data['Close'].iloc[-1]
                rsi = advanced_analytics.calculate_rsi(data).iloc[-1]
                macd = advanced_analytics.calculate_macd(data)
                volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
                price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
                
                market_sentiment = {
                    'bullish_signals': 0,
                    'bearish_signals': 0
                }
                
                # Calculate sentiment signals
                if rsi < 30: market_sentiment['bullish_signals'] += 1
                elif rsi > 70: market_sentiment['bearish_signals'] += 1
                
                if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]: market_sentiment['bullish_signals'] += 1
                else: market_sentiment['bearish_signals'] += 1
                
                if volume_ratio > 1.5 and price_change > 0: market_sentiment['bullish_signals'] += 1
                elif volume_ratio > 1.5 and price_change < 0: market_sentiment['bearish_signals'] += 1
                
                # Generate comprehensive AI decision
                ai_decision = advanced_decision_engine.generate_comprehensive_decision(
                    data, symbol, technical_signals, strategy_results, market_sentiment
                )
                
                # === DECISION SUMMARY ===
                st.subheader("🎯 AI Trading Decision Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    signal_color = "🟢" if ai_decision['overall_signal'] == 'BUY' else "🔴" if ai_decision['overall_signal'] == 'SELL' else "🟡"
                    st.metric("AI Decision", f"{signal_color} {ai_decision['overall_signal']}", 
                             f"Confidence: {ai_decision['confidence_score']:.1%}")
                
                with col2:
                    confidence_color = "success" if ai_decision['confidence_score'] >= 0.7 else "warning" if ai_decision['confidence_score'] >= 0.5 else "error"
                    st.metric("Confidence Level", ai_decision['confidence_level'], 
                             f"Score: {ai_decision['confidence_score']:.1%}")
                
                with col3:
                    risk_color = "success" if ai_decision['risk_level'] == 'LOW' else "warning" if ai_decision['risk_level'] == 'MEDIUM' else "error"
                    st.metric("Risk Assessment", ai_decision['risk_level'], 
                             f"Position: {ai_decision['position_sizing']:.1%}")
                
                with col4:
                    st.metric("Time Horizon", ai_decision['time_horizon'], 
                             f"Entry: ${ai_decision['entry_price']:.2f}")
                
                # Decision confidence check
                meets_threshold = ai_decision['confidence_score'] * 100 >= confidence_threshold
                
                if meets_threshold:
                    st.success(f"✅ Decision meets your {confidence_threshold}% confidence threshold!")
                else:
                    st.warning(f"⚠️ Decision confidence ({ai_decision['confidence_score']:.1%}) below your {confidence_threshold}% threshold. Consider waiting for better setup.")
                
                # === DETAILED ANALYSIS ===
                st.subheader("📊 Detailed AI Analysis Breakdown")
                
                tab1, tab2, tab3, tab4 = st.tabs(["🧠 AI Scoring", "📈 Entry/Exit Levels", "⚠️ Risk Analysis", "🔗 External Research"])
                
                with tab1:
                    st.write("**Multi-Factor Analysis Scores:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        analysis_breakdown = ai_decision.get('analysis_breakdown', {})
                        
                        for factor, data in analysis_breakdown.items():
                            if isinstance(data, dict) and 'score' in data:
                                score = data['score']
                                strength = data.get('strength', 'Unknown')
                                
                                score_color = "🟢" if score > 0.3 else "🔴" if score < -0.3 else "🟡"
                                st.write(f"**{factor.replace('_', ' ').title()}**: {score_color} {score:.2f} ({strength})")
                                
                                # Progress bar for visual representation
                                progress_value = (score + 1) / 2  # Convert -1,1 to 0,1
                                st.progress(progress_value)
                    
                    with col2:
                        st.write("**Supporting Factors:**")
                        for factor in ai_decision.get('supporting_factors', []):
                            st.write(f"✅ {factor}")
                        
                        if ai_decision.get('risk_factors'):
                            st.write("**Risk Factors:**")
                            for factor in ai_decision.get('risk_factors', []):
                                st.write(f"⚠️ {factor}")
                
                with tab2:
                    st.write("**Recommended Entry and Exit Levels:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Entry Price", f"${ai_decision['entry_price']:.2f}", "Current Market Price")
                        st.metric("Position Size", f"{ai_decision['position_sizing']:.1%}", "Of Portfolio")
                    
                    with col2:
                        if ai_decision['stop_loss'] != 0:
                            stop_distance = abs(ai_decision['stop_loss'] - ai_decision['entry_price']) / ai_decision['entry_price'] * 100
                            st.metric("Stop Loss", f"${ai_decision['stop_loss']:.2f}", f"{stop_distance:.1f}% away")
                        
                        if ai_decision['take_profit'] != 0:
                            profit_distance = abs(ai_decision['take_profit'] - ai_decision['entry_price']) / ai_decision['entry_price'] * 100
                            st.metric("Take Profit", f"${ai_decision['take_profit']:.2f}", f"{profit_distance:.1f}% target")
                    
                    with col3:
                        if ai_decision['stop_loss'] != 0 and ai_decision['take_profit'] != 0:
                            risk = abs(ai_decision['entry_price'] - ai_decision['stop_loss'])
                            reward = abs(ai_decision['take_profit'] - ai_decision['entry_price'])
                            risk_reward = reward / risk if risk > 0 else 0
                            st.metric("Risk/Reward Ratio", f"{risk_reward:.1f}:1", "Reward vs Risk")
                        
                        market_conditions = ai_decision.get('market_conditions', {})
                        if market_conditions.get('trend'):
                            st.metric("Market Trend", market_conditions['trend'], market_conditions.get('phase', ''))
                
                with tab3:
                    st.write("**Comprehensive Risk Assessment:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Risk Metrics:**")
                        risk_breakdown = analysis_breakdown.get('risk_score', {})
                        if risk_breakdown:
                            st.write(f"• Volatility: {risk_breakdown.get('volatility', 0):.1f}%")
                            st.write(f"• ATR Percentage: {risk_breakdown.get('atr_percentage', 0):.1f}%")
                            st.write(f"• Risk Level: {risk_breakdown.get('risk_level', 'Unknown')}")
                        
                        volume_breakdown = analysis_breakdown.get('volume_score', {})
                        if volume_breakdown:
                            st.write(f"• Volume Ratio: {volume_breakdown.get('volume_ratio', 0):.1f}x")
                            st.write(f"• Volume Trend: {volume_breakdown.get('volume_trend', 'Unknown')}")
                    
                    with col2:
                        st.write("**Risk Mitigation Strategies:**")
                        if ai_decision['confidence_score'] < 0.7:
                            st.write("⚠️ Consider reducing position size due to lower confidence")
                        if ai_decision['risk_level'] == 'HIGH':
                            st.write("⚠️ Use tighter stop losses due to high volatility")
                        if volume_breakdown.get('volume_trend') == 'Low':
                            st.write("⚠️ Monitor for volume confirmation before entry")
                        
                        st.write("✅ Diversify across multiple positions")
                        st.write("✅ Set alerts for key price levels")
                        st.write("✅ Review decision if market conditions change")
                
                with tab4:
                    st.write("**External Research and Reference Links:**")
                    st.write(f"Continue your research on **{symbol}** using these professional resources:")
                    
                    external_links = ai_decision.get('external_references', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Financial Data & Analysis:**")
                        for name, url in list(external_links.items())[:5]:
                            st.write(f"🔗 [{name}]({url})")
                    
                    with col2:
                        st.write("**News & Community:**")
                        for name, url in list(external_links.items())[5:]:
                            st.write(f"🔗 [{name}]({url})")
                    
                    st.info("💡 **Professional Tip**: Always verify AI analysis with multiple sources and your own research before making investment decisions.")
                
                # === ACTION PLAN ===
                if meets_threshold and ai_decision['overall_signal'] != 'NEUTRAL':
                    st.subheader("📋 Suggested Action Plan")
                    
                    action_steps = []
                    
                    if ai_decision['overall_signal'] == 'BUY':
                        action_steps = [
                            f"1. **Entry**: Consider buying {symbol} at current price ${ai_decision['entry_price']:.2f}",
                            f"2. **Position Size**: Allocate {ai_decision['position_sizing']:.1%} of your portfolio",
                            f"3. **Stop Loss**: Set stop loss at ${ai_decision['stop_loss']:.2f}",
                            f"4. **Take Profit**: Target profit at ${ai_decision['take_profit']:.2f}",
                            f"5. **Time Horizon**: Plan for {ai_decision['time_horizon'].lower().replace('_', ' ')} holding period",
                            "6. **Monitor**: Check external research links for any breaking news",
                            "7. **Review**: Reassess if confidence drops below your threshold"
                        ]
                    else:  # SELL
                        action_steps = [
                            f"1. **Entry**: Consider shorting or selling {symbol} at ${ai_decision['entry_price']:.2f}",
                            f"2. **Position Size**: Allocate {ai_decision['position_sizing']:.1%} of your portfolio",
                            f"3. **Stop Loss**: Set stop loss at ${ai_decision['stop_loss']:.2f}",
                            f"4. **Take Profit**: Target profit at ${ai_decision['take_profit']:.2f}",
                            f"5. **Time Horizon**: Plan for {ai_decision['time_horizon'].lower().replace('_', ' ')} holding period",
                            "6. **Monitor**: Watch external sources for any positive catalysts",
                            "7. **Review**: Reassess if market sentiment changes"
                        ]
                    
                    for step in action_steps:
                        st.write(step)
                    
                    st.warning("⚠️ **Disclaimer**: This AI analysis is for educational purposes only. Always do your own research and consider your risk tolerance before making investment decisions.")
                
            else:
                st.error("Unable to fetch data for the specified symbol. Please check the symbol and try again.")

def risk_management_interface(data_fetcher, advanced_analytics):
    """Risk management interface"""
    st.header("⚠️ Risk Management")
    st.markdown("Calculate position sizing, stop losses, and risk metrics")
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        account_size = st.number_input("Account Size ($)", value=10000, min_value=1000)
    
    with col2:
        risk_per_trade = st.slider("Risk per Trade (%)", 1, 5, 2)
        position_value = st.number_input("Position Value ($)", value=1000, min_value=100)
    
    with col3:
        timeframe = st.selectbox("Analysis Timeframe", ["1d", "1h"], index=0)
    
    if st.button("📊 Calculate Risk Metrics", type="primary"):
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        data = data_fetcher.fetch_data(symbol, timeframe, start_date=start_date, end_date=end_date)
        
        if data is not None and not data.empty:
            # Calculate risk metrics
            risk_metrics = advanced_analytics.calculate_risk_metrics(data, position_value)
            
            # Display risk analysis
            st.subheader("📋 Risk Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Recommended Position Size", f"{risk_metrics['position_size_shares']} shares")
                st.metric("Stop Loss Level", f"${risk_metrics['stop_loss_level']:.2f}")
                st.metric("Take Profit Level", f"${risk_metrics['take_profit_level']:.2f}")
                
                # Risk/Reward calculation
                current_price = data['Close'].iloc[-1]
                risk_amount = current_price - risk_metrics['stop_loss_level']
                reward_amount = risk_metrics['take_profit_level'] - current_price
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
                st.metric("Risk/Reward Ratio", f"1:{risk_reward_ratio:.1f}")
            
            with col2:
                st.metric("Daily Volatility", f"{risk_metrics['daily_volatility']:.1%}")
                st.metric("Current ATR", f"${risk_metrics['current_atr']:.2f}")
                st.metric("Max Historical Drawdown", f"{risk_metrics['max_drawdown']:.1%}")
                
                # Portfolio risk
                portfolio_risk = (position_value / account_size) * 100
                st.metric("Portfolio Exposure", f"{portfolio_risk:.1%}")
            
            # Risk warnings
            st.subheader("⚠️ Risk Warnings")
            
            if portfolio_risk > 10:
                st.error("HIGH RISK: Position size exceeds 10% of account")
            elif portfolio_risk > 5:
                st.warning("MODERATE RISK: Position size exceeds 5% of account")
            else:
                st.success("ACCEPTABLE RISK: Position size within safe limits")
            
            if risk_reward_ratio < 1.5:
                st.warning("Risk/Reward ratio below 1.5:1 - Consider adjusting targets")
            
            # Position sizing table
            st.subheader("📊 Position Sizing Options")
            
            risk_levels = [1, 2, 3, 5]
            sizing_data = []
            
            for risk_pct in risk_levels:
                max_loss = account_size * (risk_pct / 100)
                shares = int(max_loss / risk_metrics['risk_per_share']) if risk_metrics['risk_per_share'] > 0 else 0
                position_val = shares * current_price
                
                sizing_data.append({
                    'Risk %': f"{risk_pct}%",
                    'Max Loss': f"${max_loss:.0f}",
                    'Shares': shares,
                    'Position Value': f"${position_val:.0f}",
                    'Stop Loss': f"${risk_metrics['stop_loss_level']:.2f}"
                })
            
            st.dataframe(pd.DataFrame(sizing_data), use_container_width=True)
            
        else:
            st.error("Could not fetch data for the specified symbol.")

def trading_advisor_interface(data_fetcher, pattern_recognition, advanced_analytics, decision_engine):
    """Trading advisor interface for immediate buy/sell decisions"""
    st.header("🎯 Trading Advisor")
    st.markdown("Get immediate trading recommendations with confidence levels and risk analysis")
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        account_size = st.number_input("Account Size ($)", value=10000, min_value=1000)
    
    with col2:
        timeframe = st.selectbox("Analysis Timeframe", ["1d", "1h", "15min"], index=0)
        analysis_mode = st.selectbox("Analysis Depth", ["Quick Analysis", "Deep Analysis"], index=1)
    
    with col3:
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    if st.button("🚀 Get Trading Recommendation", type="primary"):
        with st.spinner("Analyzing market conditions and generating recommendation..."):
            
            # Fetch data
            data = data_fetcher.fetch_data(symbol, timeframe, period="2mo")
            
            if data is not None and not data.empty:
                # Analyze patterns
                patterns = pattern_recognition.analyze_patterns(data, pattern_recognition.patterns.keys())
                
                # Generate trading decision
                decision = decision_engine.make_trading_decision(
                    symbol, data, patterns, advanced_analytics, account_size
                )
                
                # Display recommendation prominently
                st.markdown("---")
                
                # Main recommendation card
                if decision.action in ['BUY', 'WEAK_BUY']:
                    action_color = "green"
                    action_emoji = "📈"
                elif decision.action in ['SELL', 'WEAK_SELL']:
                    action_color = "red"
                    action_emoji = "📉"
                else:
                    action_color = "orange"
                    action_emoji = "⏸️"
                
                st.markdown(f"""
                <div style="background-color: {action_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <h2 style="color: white; margin: 0;">{action_emoji} {decision.action}</h2>
                    <h3 style="color: white; margin: 10px 0;">Confidence: {decision.confidence:.1%}</h3>
                    <h3 style="color: white; margin: 10px 0;">Urgency: {decision.urgency}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    st.metric("Entry Price", f"${decision.entry_price:.2f}", f"Date: {current_date}")
                    st.metric("Position Size", f"{decision.position_size:.1%} of account")
                
                with col2:
                    st.metric("Stop Loss", f"${decision.stop_loss:.2f}")
                    st.metric("Max Risk", f"{decision.max_risk:.1%}")
                
                with col3:
                    st.metric("Take Profit", f"${decision.take_profit:.2f}")
                    st.metric("Expected Return", f"{decision.expected_return:.1%}")
                
                with col4:
                    st.metric("Risk/Reward", f"1:{decision.risk_reward_ratio:.1f}")
                    st.metric("Timeframe", decision.timeframe)
                
                # Detailed reasoning
                st.markdown("---")
                st.subheader("📝 Decision Reasoning")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Supporting Signals:**")
                    for signal in decision.signals_supporting:
                        st.write(f"• {signal}")
                    
                    st.markdown("**🎯 Main Reasons:**")
                    for reason in decision.reasoning:
                        st.write(f"• {reason}")
                
                with col2:
                    if decision.signals_against:
                        st.markdown("**⚠️ Contrary Signals:**")
                        for signal in decision.signals_against:
                            st.write(f"• {signal}")
                    else:
                        st.markdown("**✅ No significant contrary signals detected**")
                
                # Action plan
                st.markdown("---")
                st.subheader("📋 Action Plan")
                
                if decision.action == 'BUY':
                    st.success(f"""
                    **Immediate Action Required:**
                    1. Place BUY order for {symbol} at market price (${decision.entry_price:.2f})
                    2. Set stop loss at ${decision.stop_loss:.2f}
                    3. Set take profit at ${decision.take_profit:.2f}
                    4. Position size: {decision.position_size:.1%} of account
                    5. Monitor position closely for {decision.timeframe.lower()}
                    """)
                
                elif decision.action == 'SELL':
                    st.error(f"""
                    **Immediate Action Required:**
                    1. Place SELL order for {symbol} at market price (${decision.entry_price:.2f})
                    2. Set stop loss at ${decision.stop_loss:.2f}
                    3. Set take profit at ${decision.take_profit:.2f}
                    4. Position size: {decision.position_size:.1%} of account
                    5. Monitor position closely for {decision.timeframe.lower()}
                    """)
                
                elif decision.action in ['WEAK_BUY', 'WEAK_SELL']:
                    st.warning(f"""
                    **Consider Action (Lower Confidence):**
                    1. Wait for additional confirmation signals
                    2. Consider smaller position size: {decision.position_size * 0.5:.1%} of account
                    3. Monitor closely for trend strengthening
                    4. Entry price: ${decision.entry_price:.2f}
                    """)
                
                else:
                    st.info(f"""
                    **Hold/Wait Recommendation:**
                    1. Current signals are mixed or neutral
                    2. Wait for clearer market direction
                    3. Set alerts for breakout above/below key levels
                    4. Review again in 1-2 trading sessions
                    """)
                
                # Risk warnings
                if decision.max_risk > 0.05:  # More than 5% risk
                    st.error("⚠️ HIGH RISK WARNING: This trade risks more than 5% of your account!")
                
                if decision.urgency == 'CRITICAL':
                    st.error("🚨 CRITICAL TIMING: Execute this trade immediately or opportunity may be lost!")
                
            else:
                st.error("Could not fetch data for the specified symbol.")

def backtesting_interface(advanced_backtesting, pattern_recognition):
    """Advanced backtesting interface"""
    st.header("⏮️ Advanced Backtesting")
    st.markdown("Test different trading strategies on historical data with comprehensive metrics")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", value=datetime.now())
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "1h"], index=0)
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000)
    
    # Strategy selection
    st.subheader("📊 Strategy Selection")
    strategies = st.multiselect(
        "Select strategies to test",
        ["Pattern Strategy", "Technical Strategy", "Combined Strategy", "Risk-Adjusted Strategy"],
        default=["Combined Strategy", "Risk-Adjusted Strategy"]
    )
    
    if st.button("🔬 Run Backtest", type="primary"):
        if not strategies:
            st.error("Please select at least one strategy to test.")
            return
        
        with st.spinner("Running comprehensive backtest... This may take a few minutes."):
            
            # Run backtest
            results = advanced_backtesting.run_comprehensive_backtest(
                symbol, start_date, end_date, pattern_recognition, timeframe
            )
            
            if 'error' in results:
                st.error(f"Backtest failed: {results['error']}")
                return
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Backtest Results")
            
            # Strategy comparison table
            comparison_data = []
            for strategy_name in strategies:
                strategy_key = strategy_name.lower().replace(' ', '_').replace('-', '_')
                if strategy_key in results:
                    result = results[strategy_key]
                    comparison_data.append({
                        'Strategy': strategy_name,
                        'Total Return': f"{result.total_return:.1%}",
                        'Win Rate': f"{result.win_rate:.1%}",
                        'Total Trades': result.total_trades,
                        'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                        'Max Drawdown': f"{result.max_drawdown:.1%}",
                        'Profit Factor': f"{result.profit_factor:.2f}"
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Best strategy highlight
                best_strategy = results.get('strategy_comparison', {}).get('best_strategy', 'Unknown')
                if best_strategy in [s.lower().replace(' ', '_').replace('-', '_') for s in strategies]:
                    st.success(f"🏆 Best performing strategy: {best_strategy.replace('_', ' ').title()}")
            
            # Detailed strategy results
            for strategy_name in strategies:
                strategy_key = strategy_name.lower().replace(' ', '_').replace('-', '_')
                if strategy_key in results:
                    result = results[strategy_key]
                    
                    with st.expander(f"📊 {strategy_name} - Detailed Results"):
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Return", f"{result.total_return:.1%}")
                            st.metric("Average Return", f"{result.avg_return:.2%}")
                        
                        with col2:
                            st.metric("Win Rate", f"{result.win_rate:.1%}")
                            st.metric("Total Trades", result.total_trades)
                        
                        with col3:
                            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                            st.metric("Calmar Ratio", f"{result.calmar_ratio:.2f}")
                        
                        with col4:
                            st.metric("Max Drawdown", f"{result.max_drawdown:.1%}")
                            st.metric("Profit Factor", f"{result.profit_factor:.2f}")
                        
                        # Advanced metrics
                        st.markdown("**Advanced Metrics:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"• Best Trade: {result.best_trade:.1%}")
                            st.write(f"• Worst Trade: {result.worst_trade:.1%}")
                            st.write(f"• Avg Winning Trade: {result.avg_winning_return:.1%}")
                        
                        with col2:
                            st.write(f"• Avg Losing Trade: {result.avg_losing_return:.1%}")
                            st.write(f"• Max Consecutive Wins: {result.max_consecutive_wins}")
                            st.write(f"• Max Consecutive Losses: {result.max_consecutive_losses}")
                        
                        # Trade history (last 10 trades)
                        if result.trades:
                            st.markdown("**Recent Trades:**")
                            recent_trades = result.trades[-10:]
                            trade_data = []
                            
                            for trade in recent_trades:
                                trade_data.append({
                                    'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                                    'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                                    'Entry Price': f"${trade.entry_price:.2f}",
                                    'Exit Price': f"${trade.exit_price:.2f}",
                                    'Type': trade.trade_type.upper(),
                                    'Pattern': trade.pattern_type,
                                    'Return': f"{trade.return_pct:.1%}",
                                    'Holding Days': trade.holding_days,
                                    'Confidence': f"{trade.confidence:.1%}"
                                })
                            
                            df_trades = pd.DataFrame(trade_data)
                            st.dataframe(df_trades, use_container_width=True)
            
            # Monte Carlo simulation results
            if 'monte_carlo' in results and 'error' not in results['monte_carlo']:
                st.markdown("---")
                st.subheader("🎲 Monte Carlo Simulation")
                
                mc_results = results['monte_carlo']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean Return", f"{mc_results['mean_return']:.1%}")
                    st.metric("Probability of Profit", f"{mc_results['probability_profit']:.1%}")
                
                with col2:
                    st.metric("95% Confidence Range", 
                             f"{mc_results['confidence_95_lower']:.1%} to {mc_results['confidence_95_upper']:.1%}")
                
                with col3:
                    st.metric("Best Case", f"{mc_results['max_simulated_return']:.1%}")
                    st.metric("Worst Case", f"{mc_results['min_simulated_return']:.1%}")
                
                # Risk assessment
                st.markdown("**Risk Assessment:**")
                if mc_results['probability_profit'] > 0.6:
                    st.success("High probability of profit - Strategy shows strong potential")
                elif mc_results['probability_profit'] > 0.5:
                    st.warning("Moderate probability of profit - Consider risk management")
                else:
                    st.error("Low probability of profit - Strategy needs improvement")

def portfolio_interface(portfolio_tracker, data_fetcher, advanced_analytics):
    """Portfolio tracking interface"""
    st.header("💼 Portfolio Tracker")
    st.markdown("Manage and analyze your investment portfolio with professional-grade metrics")
    
    # Portfolio actions
    st.subheader("📊 Portfolio Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Add Position**")
        
        # Simple form without session state conflicts
        symbol = st.text_input("Symbol", placeholder="e.g., AAPL", key="portfolio_symbol").upper()
        quantity = st.number_input("Shares", min_value=1, value=100, key="portfolio_quantity")
        avg_cost = st.number_input("Average Cost ($)", min_value=0.01, value=100.0, key="portfolio_cost")
        
        if st.button("Add Position", type="primary", key="add_position_btn"):
            if symbol and len(symbol) > 0:
                if symbol not in portfolio_tracker.positions:
                    portfolio_tracker.add_position(symbol, quantity, avg_cost)
                    st.success(f"✅ Added {quantity} shares of {symbol} at ${avg_cost:.2f}")
                    st.rerun()
                else:
                    st.error(f"❌ {symbol} already exists in portfolio. Remove it first to update.")
            else:
                st.error("❌ Please enter a valid symbol")
    
    with col2:
        st.markdown("**Remove Position**")
        if portfolio_tracker.positions:
            symbols = list(portfolio_tracker.positions.keys())
            remove_symbol = st.selectbox("Select position to remove", ["Select symbol..."] + symbols, key="remove_symbol")
            
            if st.button("Remove Position", key="remove_btn"):
                if remove_symbol and remove_symbol != "Select symbol...":
                    portfolio_tracker.remove_position(remove_symbol)
                    st.success(f"✅ Removed {remove_symbol} from portfolio")
                    st.rerun()
                else:
                    st.error("❌ Please select a symbol to remove")
        else:
            st.info("💼 No positions in portfolio yet")
    
    # Portfolio overview
    if portfolio_tracker.positions:
        st.markdown("---")
        st.subheader("📈 Portfolio Overview")
        
        with st.spinner("Calculating portfolio metrics..."):
            metrics = portfolio_tracker.get_portfolio_metrics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${metrics.total_value:,.2f}")
            st.metric("Total P&L", f"${metrics.total_pnl:,.2f}", f"{metrics.total_pnl_pct:.1%}")
        
        with col2:
            st.metric("Daily P&L", f"${metrics.daily_pnl:,.2f}", f"{metrics.daily_pnl_pct:.1%}")
            st.metric("Beta", f"{metrics.beta:.2f}")
        
        with col3:
            st.metric("Alpha", f"{metrics.alpha:.1%}")
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        
        with col4:
            st.metric("Volatility", f"{metrics.volatility:.1%}")
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.1%}")
        
        # Positions table
        st.subheader("📋 Current Positions")
        
        positions_data = []
        for position in portfolio_tracker.positions.values():
            positions_data.append({
                'Symbol': position.symbol,
                'Shares': f"{position.quantity:,.0f}",
                'Avg Cost': f"${position.avg_cost:.2f}",
                'Current Price': f"${position.current_price:.2f}",
                'Market Value': f"${position.market_value:,.2f}",
                'P&L': f"${position.unrealized_pnl:,.2f}",
                'P&L %': f"{position.unrealized_pnl_pct:.1%}",
                'Weight': f"{position.weight:.1%}",
                'Sector': position.sector
            })
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True)
        
        # Sector allocation
        st.subheader("🏭 Sector Allocation")
        sector_allocation = portfolio_tracker.get_sector_allocation()
        
        if sector_allocation:
            sector_df = pd.DataFrame(list(sector_allocation.items()), 
                                   columns=['Sector', 'Allocation'])
            sector_df['Allocation %'] = sector_df['Allocation'].apply(lambda x: f"{x:.1%}")
            st.dataframe(sector_df, use_container_width=True)
    
    else:
        st.info("Add positions to your portfolio to see analytics")

def options_interface(options_analyzer, data_fetcher):
    """Options analysis interface"""
    st.header("📊 Options Analyzer")
    st.markdown("Comprehensive options analysis with Greeks, strategies, and arbitrage detection")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", key="options_symbol").upper()
        investment_amount = st.number_input("Investment Amount ($)", value=10000, min_value=1000)
    
    with col2:
        analysis_type = st.selectbox("Analysis Type", 
                                   ["Option Chain", "Strategy Analysis", "IV Analysis"])
    
    if st.button("🔍 Analyze Options", type="primary"):
        with st.spinner("Fetching options data and analyzing..."):
            
            if analysis_type == "Option Chain":
                # Get option chain
                option_data = options_analyzer.get_option_chain(symbol)
                
                if 'error' in option_data:
                    st.error(f"Error: {option_data['error']}")
                    return
                
                # Display option chain
                st.subheader(f"📈 {symbol} Option Chain")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${option_data['current_price']:.2f}")
                with col2:
                    st.metric("Total Call Volume", f"{option_data['total_call_volume']:,}")
                with col3:
                    st.metric("Put/Call Ratio", f"{option_data['put_call_ratio']:.2f}")
                
                # Show available expiries
                st.write(f"**Available Expiries:** {', '.join(option_data['available_expiries'][:5])}")
                
                # Show summary of options
                st.success(f"Found {len(option_data['calls'])} call options and {len(option_data['puts'])} put options")
            
            elif analysis_type == "Strategy Analysis":
                # Options strategy analysis
                st.subheader(f"📊 {symbol} Options Strategy Analysis")
                
                # Get current stock data for strategy analysis
                data = data_fetcher.fetch_data(symbol, '1d', period='1mo')
                current_price = data['Close'].iloc[-1]
                volatility = data['Close'].pct_change().std() * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("30-day Volatility", f"{volatility:.1%}")
                
                with col2:
                    # Strategy recommendations based on market conditions
                    if volatility > 0.03:  # High volatility
                        recommended_strategy = "Iron Condor (Sell volatility)"
                        strategy_reason = "High volatility environment favors volatility selling strategies"
                    else:
                        recommended_strategy = "Long Straddle (Buy volatility)"
                        strategy_reason = "Low volatility environment favors volatility buying strategies"
                    
                    st.success(f"**Recommended:** {recommended_strategy}")
                    st.info(f"**Reason:** {strategy_reason}")
                
                # Strategy comparison table
                st.markdown("**Options Strategy Comparison:**")
                
                strategies_data = [
                    {"Strategy": "Covered Call", "Market View": "Neutral to Bullish", "Max Profit": "Limited", "Max Loss": "Limited", "Complexity": "Low"},
                    {"Strategy": "Cash-Secured Put", "Market View": "Neutral to Bullish", "Max Profit": "Limited", "Max Loss": "Substantial", "Complexity": "Low"},
                    {"Strategy": "Long Straddle", "Market View": "High Volatility", "Max Profit": "Unlimited", "Max Loss": "Premium Paid", "Complexity": "Medium"},
                    {"Strategy": "Iron Condor", "Market View": "Low Volatility", "Max Profit": "Net Premium", "Max Loss": "Limited", "Complexity": "High"},
                    {"Strategy": "Bull Call Spread", "Market View": "Moderately Bullish", "Max Profit": "Limited", "Max Loss": "Limited", "Complexity": "Medium"}
                ]
                
                df_strategies = pd.DataFrame(strategies_data)
                st.dataframe(df_strategies, use_container_width=True)
                
                # Profit/Loss visualization
                st.markdown("**Strategy P&L Analysis:**")
                
                # Calculate potential profits for different strategies
                price_range = [current_price * (1 + i * 0.1) for i in range(-5, 6)]
                
                covered_call_pl = [min(price - current_price, 5) for price in price_range]
                long_straddle_pl = [abs(price - current_price) - 5 for price in price_range]
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_range, y=covered_call_pl, name="Covered Call", line=dict(color='green')))
                fig.add_trace(go.Scatter(x=price_range, y=long_straddle_pl, name="Long Straddle", line=dict(color='blue')))
                
                fig.update_layout(
                    title="Options Strategy P&L Comparison",
                    xaxis_title="Stock Price at Expiration",
                    yaxis_title="Profit/Loss (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "IV Analysis":
                # Implied Volatility Analysis
                st.subheader(f"📈 {symbol} Implied Volatility Analysis")
                
                # Get stock data for IV analysis
                data = data_fetcher.fetch_data(symbol, '1d', period='3mo')
                current_price = data['Close'].iloc[-1]
                
                # Calculate historical volatility
                returns = data['Close'].pct_change().dropna()
                historical_vol_30d = returns.tail(30).std() * (252**0.5) * 100
                historical_vol_90d = returns.tail(90).std() * (252**0.5) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("30-Day HV", f"{historical_vol_30d:.1f}%")
                
                with col3:
                    st.metric("90-Day HV", f"{historical_vol_90d:.1f}%")
                
                # IV analysis insights
                st.markdown("**Volatility Analysis:**")
                
                if historical_vol_30d > historical_vol_90d * 1.2:
                    vol_regime = "High Volatility"
                    vol_color = "red"
                    trading_advice = "Consider volatility selling strategies (Iron Condors, Covered Calls)"
                elif historical_vol_30d < historical_vol_90d * 0.8:
                    vol_regime = "Low Volatility"
                    vol_color = "green"
                    trading_advice = "Consider volatility buying strategies (Long Straddles, Long Strangles)"
                else:
                    vol_regime = "Normal Volatility"
                    vol_color = "orange"
                    trading_advice = "Consider directional strategies based on technical analysis"
                
                st.markdown(f"<div style='color: {vol_color}'>**Current Regime:** {vol_regime}</div>", unsafe_allow_html=True)
                st.info(f"**Trading Advice:** {trading_advice}")
                
                # Volatility chart
                vol_data = returns.rolling(30).std() * (252**0.5) * 100
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vol_data.index,
                    y=vol_data.values,
                    mode='lines',
                    name='30-Day Historical Volatility',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title=f"{symbol} Historical Volatility Trend",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Options trading recommendations
                st.markdown("**Options Trading Recommendations:**")
                
                recommendations = [
                    f"**High Probability:** Sell options when IV > {historical_vol_90d * 1.2:.1f}%",
                    f"**High Probability:** Buy options when IV < {historical_vol_90d * 0.8:.1f}%",
                    f"**Current Volatility Rank:** {((historical_vol_30d - historical_vol_90d) / historical_vol_90d * 100):.1f}%"
                ]
                
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            else:
                st.error("Please select a valid analysis type")

def trading_intelligence_interface(data_fetcher, advanced_analytics, pattern_recognition, 
                                 trading_intelligence, alert_system):
    """Advanced Trading Intelligence Interface"""
    st.header("🧠 Trading Intelligence")
    st.markdown("**Professional AI-powered market analysis with confidence scoring and comprehensive insights**")
    
    # Market Intelligence Overview
    with st.spinner("Analyzing market conditions..."):
        market_intel = trading_intelligence.analyze_market_intelligence()
    
    # Market Overview Dashboard
    st.subheader("📊 Market Intelligence Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        regime_color = "green" if "BULL" in market_intel.market_regime else "red" if "BEAR" in market_intel.market_regime else "orange"
        st.markdown(f"<div style='color: {regime_color}'>**Market Regime:** {market_intel.market_regime.replace('_', ' ')}</div>", unsafe_allow_html=True)
    
    with col2:
        vol_color = "red" if "HIGH" in market_intel.volatility_regime else "green" if "LOW" in market_intel.volatility_regime else "orange"
        st.markdown(f"<div style='color: {vol_color}'>**Volatility:** {market_intel.volatility_regime.replace('_', ' ')}</div>", unsafe_allow_html=True)
    
    with col3:
        trend_color = "green" if market_intel.trend_strength > 60 else "red" if market_intel.trend_strength < 40 else "orange"
        st.markdown(f"<div style='color: {trend_color}'>**Trend Strength:** {market_intel.trend_strength:.0f}/100</div>", unsafe_allow_html=True)
    
    with col4:
        sentiment_color = "green" if "BULL" in market_intel.market_sentiment else "red" if "BEAR" in market_intel.market_sentiment else "orange"
        st.markdown(f"<div style='color: {sentiment_color}'>**Sentiment:** {market_intel.market_sentiment}</div>", unsafe_allow_html=True)
    
    # Key Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Key Market Insights:**")
        for insight in market_intel.key_insights:
            st.write(f"• {insight}")
    
    with col2:
        st.markdown("**⚠️ Market Risks:**")
        for risk in market_intel.market_risks:
            st.write(f"• {risk}")
    
    # Recommended Allocation
    st.markdown("**💼 Recommended Asset Allocation:**")
    allocation_data = list(market_intel.recommended_allocation.items())
    
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Pie(
        labels=[item[0] for item in allocation_data],
        values=[item[1] for item in allocation_data],
        hole=0.3,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title="Strategic Asset Allocation",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual Stock Analysis
    st.markdown("---")
    st.subheader("🎯 Individual Stock Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", key="intel_symbol").upper()
        
    with col2:
        analysis_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    if st.button("🔬 Generate Trading Intelligence", type="primary"):
        with st.spinner(f"Analyzing {symbol} with AI trading intelligence..."):
            
            try:
                # Fetch data and run analysis
                data = data_fetcher.fetch_data(symbol, '1d', period=analysis_period)
                patterns = pattern_recognition.analyze_patterns(data, list(pattern_recognition.patterns.keys()))
                technical_analysis = advanced_analytics.calculate_technical_indicators(data)
                
                # Generate AI trading signal
                trading_signal = trading_intelligence.generate_trading_signal(
                    symbol, data, patterns, technical_analysis
                )
                
                # Display comprehensive analysis
                st.subheader(f"🤖 AI Trading Signal for {symbol}")
                
                # Signal Header
                signal_color = "green" if trading_signal.signal_type == "BUY" else "red" if trading_signal.signal_type == "SELL" else "orange"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"<div style='color: {signal_color}; font-size: 24px; font-weight: bold'>{trading_signal.signal_type}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Grade:** {trading_signal.trade_grade}")
                
                with col2:
                    st.metric("Confidence", f"{trading_signal.confidence:.1f}%")
                    st.metric("Success Probability", f"{trading_signal.probability_success:.1f}%")
                
                with col3:
                    st.metric("Expected Return", f"{trading_signal.expected_return:.1f}%")
                    st.metric("Max Risk", f"{trading_signal.max_risk:.1f}%")
                
                # Price Targets
                st.markdown("**💰 Price Targets & Risk Management:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Entry Price", f"${trading_signal.entry_price:.2f}")
                
                with col2:
                    st.metric("Target Price", f"${trading_signal.target_price:.2f}")
                
                with col3:
                    st.metric("Stop Loss", f"${trading_signal.stop_loss:.2f}")
                
                # Trading Rationale
                st.markdown("**📝 AI Trading Rationale:**")
                st.info(trading_signal.rationale)
                
                # Supporting & Risk Factors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Supporting Factors:**")
                    for factor in trading_signal.supporting_factors:
                        st.write(f"• {factor}")
                
                with col2:
                    st.markdown("**⚠️ Risk Factors:**")
                    for factor in trading_signal.risk_factors:
                        st.write(f"• {factor}")
                
                # Risk-Reward Visualization
                st.markdown("**📊 Risk-Reward Analysis:**")
                
                # Calculate risk-reward metrics
                risk_amount = abs(trading_signal.entry_price - trading_signal.stop_loss)
                reward_amount = abs(trading_signal.target_price - trading_signal.entry_price)
                risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Amount", f"${risk_amount:.2f}")
                
                with col2:
                    st.metric("Reward Amount", f"${reward_amount:.2f}")
                
                with col3:
                    ratio_color = "green" if risk_reward_ratio >= 2 else "orange" if risk_reward_ratio >= 1.5 else "red"
                    st.markdown(f"<div style='color: {ratio_color}'>**R:R Ratio:** {risk_reward_ratio:.1f}:1</div>", unsafe_allow_html=True)
                
                # Action Plan
                st.markdown("**🎯 Recommended Action Plan:**")
                
                if trading_signal.confidence >= 75:
                    action_color = "green"
                    action_text = "HIGH CONFIDENCE - Consider taking action"
                elif trading_signal.confidence >= 50:
                    action_color = "orange"
                    action_text = "MODERATE CONFIDENCE - Proceed with caution"
                else:
                    action_color = "red"
                    action_text = "LOW CONFIDENCE - Wait for better setup"
                
                st.markdown(f"<div style='color: {action_color}; font-weight: bold'>{action_text}</div>", unsafe_allow_html=True)
                
                # Professional Trading Notes
                st.markdown("**📋 Professional Notes:**")
                st.info(f"""
                **Entry Strategy:** {trading_signal.signal_type} {symbol} at ${trading_signal.entry_price:.2f}
                
                **Risk Management:** Set stop-loss at ${trading_signal.stop_loss:.2f} for {trading_signal.max_risk:.1f}% maximum risk
                
                **Profit Target:** Target ${trading_signal.target_price:.2f} for {trading_signal.expected_return:.1f}% potential return
                
                **Time Horizon:** {trading_signal.time_horizon.replace('_', ' ').title()} term strategy recommended
                
                **Trade Grade:** {trading_signal.trade_grade} - {trading_signal.probability_success:.0f}% probability of success
                """)
                
                st.warning("💡 To set price alerts for these levels, use the dedicated Alert System tab")
                
                # Position Sizing Recommendation
                st.markdown("**📏 Position Sizing Recommendation:**")
                
                portfolio_value = st.number_input("Portfolio Value ($)", value=100000, min_value=1000)
                max_risk_pct = st.slider("Max Risk per Trade (%)", min_value=1, max_value=5, value=2)
                
                risk_per_share = abs(trading_signal.entry_price - trading_signal.stop_loss)
                max_risk_amount = portfolio_value * (max_risk_pct / 100)
                recommended_shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
                position_value = recommended_shares * trading_signal.entry_price
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Recommended Shares", f"{recommended_shares:,}")
                
                with col2:
                    st.metric("Position Value", f"${position_value:,.0f}")
                
                with col3:
                    portfolio_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                    st.metric("Portfolio %", f"{portfolio_pct:.1f}%")
                
            except Exception as e:
                st.error(f"Error in trading intelligence analysis: {str(e)}")
                logger.error(f"Trading intelligence error: {e}")

def alert_interface(alert_system, data_fetcher, pattern_recognition, advanced_analytics):
    """Alert system interface"""
    st.header("🚨 Alert System")
    st.markdown("Set up intelligent alerts for price movements, patterns, and technical signals")
    
    # Alert management tabs
    tab1, tab2, tab3 = st.tabs(["Create Alerts", "Active Alerts", "Alert History"])
    
    with tab1:
        st.subheader("🔔 Create New Alert")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", value="AAPL", key="alert_symbol").upper()
            alert_type = st.selectbox("Alert Type", 
                                    ["Price Breakout", "Volume Spike", "Pattern Formation", 
                                     "Technical Signal", "Smart Alerts"])
        
        with col2:
            priority = st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        
        if alert_type == "Price Breakout":
            target_price = st.number_input("Target Price ($)", value=150.0, min_value=0.01)
            direction = st.selectbox("Direction", ["above", "below"])
            
            if st.button("Create Price Alert", type="primary"):
                alert_id = alert_system.create_price_alert(symbol, target_price, direction)
                st.success(f"Created price alert {alert_id[-8:]} for {symbol}")
                st.session_state.alert_system = alert_system  # Ensure persistence
        
        elif alert_type == "Volume Spike":
            volume_multiplier = st.number_input("Volume Multiplier", value=2.0, min_value=1.1)
            
            if st.button("Create Volume Alert", type="primary"):
                alert_id = alert_system.create_volume_alert(symbol, volume_multiplier)
                st.success(f"Created volume alert {alert_id[-8:]} for {symbol}")
                st.session_state.alert_system = alert_system
        
        elif alert_type == "Pattern Formation":
            patterns = st.multiselect("Patterns to Watch", 
                                    ["Hammer", "Doji", "Engulfing", "Morning Star", "Evening Star"])
            
            if st.button("Create Pattern Alert", type="primary") and patterns:
                alert_id = alert_system.create_pattern_alert(symbol, patterns)
                st.success(f"Created pattern alert {alert_id[-8:]} for {symbol}")
                st.session_state.alert_system = alert_system
        
        elif alert_type == "Technical Signal":
            indicator = st.selectbox("Indicator", ["RSI", "MACD", "Stochastic"])
            value = st.number_input("Threshold Value", value=70.0)
            operator = st.selectbox("Condition", [">=", "<=", ">", "<"])
            
            if st.button("Create Technical Alert", type="primary"):
                alert_id = alert_system.create_technical_alert(symbol, indicator, value, operator)
                st.success(f"Created technical alert {alert_id[-8:]} for {symbol}")
                st.session_state.alert_system = alert_system
        
        elif alert_type == "Smart Alerts":
            if st.button("Create Smart Alerts", type="primary"):
                with st.spinner("Analyzing market conditions and creating intelligent alerts..."):
                    alert_ids = alert_system.create_smart_alerts_for_symbol(
                        symbol, data_fetcher, pattern_recognition, advanced_analytics
                    )
                st.session_state.alert_system = alert_system
                st.success(f"Created {len(alert_ids)} smart alerts for {symbol}")
        
        # Bulk alert creation
        st.markdown("---")
        st.subheader("📦 Bulk Alert Creation")
        
        bulk_symbols = st.text_area(
            "Multiple Symbols (comma-separated)", 
            value="AAPL,MSFT,GOOGL",
            help="Enter multiple symbols to create alerts for all at once"
        )
        
        bulk_alert_type = st.selectbox(
            "Bulk Alert Type", 
            ["Price Breakout", "Volume Spike", "Smart Alerts"],
            key="bulk_type"
        )
        
        if bulk_alert_type == "Price Breakout":
            bulk_price = st.number_input("Target Price ($)", value=200.0, key="bulk_price")
            bulk_direction = st.selectbox("Direction", ["above", "below"], key="bulk_direction")
        elif bulk_alert_type == "Volume Spike":
            bulk_volume = st.number_input("Volume Multiplier", value=2.0, key="bulk_volume")
        
        if st.button("🚀 Create Bulk Alerts", type="secondary"):
            symbols_list = [s.strip().upper() for s in bulk_symbols.split(',') if s.strip()]
            created_count = 0
            
            with st.spinner(f"Creating {bulk_alert_type} alerts for {len(symbols_list)} symbols..."):
                for sym in symbols_list:
                    try:
                        if bulk_alert_type == "Price Breakout":
                            alert_system.create_price_alert(sym, bulk_price, bulk_direction)
                        elif bulk_alert_type == "Volume Spike":
                            alert_system.create_volume_alert(sym, bulk_volume)
                        elif bulk_alert_type == "Smart Alerts":
                            alert_system.create_smart_alerts_for_symbol(
                                sym, data_fetcher, pattern_recognition, advanced_analytics
                            )
                        created_count += 1
                    except Exception as e:
                        st.warning(f"Failed to create alert for {sym}: {str(e)}")
            
            st.session_state.alert_system = alert_system
            st.success(f"Successfully created {bulk_alert_type} alerts for {created_count} symbols")
            st.rerun()
    
    with tab2:
        st.subheader("📋 Active Alerts")
        
        active_alerts = alert_system.get_active_alerts()
        
        if active_alerts:
            st.write(f"You have {len(active_alerts)} active alerts")
            
            # Display all alerts with their unique IDs
            for i, alert in enumerate(active_alerts):
                priority_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                emoji = priority_emoji.get(alert.priority, "⚪")
                
                with st.expander(f"{emoji} {alert.symbol} - {alert.alert_type} ({alert.priority}) - ID: {alert.id[-8:]}"):
                    st.write(f"**Condition:** {alert.condition}")
                    st.write(f"**Target:** ${alert.target_value:.2f}")
                    st.write(f"**Current:** ${alert.current_value:.2f}")
                    st.write(f"**Created:** {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Confidence:** {alert.confidence:.1%}")
                    
                    # Remove alert button
                    if st.button(f"Remove Alert", key=f"remove_{alert.id}"):
                        alert_system.remove_alert(alert.id)
                        st.success("Alert removed")
                        st.rerun()
        else:
            st.info("No active alerts")
        
        # Check alerts button
        if st.button("🔍 Check All Alerts", type="primary"):
            with st.spinner("Checking all active alerts..."):
                triggered = alert_system.check_alerts(data_fetcher, pattern_recognition, advanced_analytics)
            
            if triggered:
                st.success(f"Found {len(triggered)} triggered alerts!")
                for alert in triggered:
                    st.warning(f"🚨 ALERT: {alert.message}")
            else:
                st.info("No alerts triggered")
    
    with tab3:
        st.subheader("📊 Alert Statistics")
        
        # Alert statistics
        stats = alert_system.get_alert_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", stats['total_alerts'])
        with col2:
            st.metric("Active Alerts", stats['active_alerts'])
        with col3:
            st.metric("Triggered Alerts", stats['triggered_alerts'])
        with col4:
            st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
        
        # Recent triggered alerts
        st.subheader("📈 Recent Alert History")
        triggered_alerts = alert_system.get_triggered_alerts(hours=168)  # Last week
        
        if triggered_alerts:
            alert_data = []
            for alert in triggered_alerts[-10:]:  # Last 10 triggered
                alert_data.append({
                    'Symbol': alert.symbol,
                    'Type': alert.alert_type,
                    'Triggered': alert.triggered_at.strftime('%Y-%m-%d %H:%M'),
                    'Condition': alert.condition,
                    'Priority': alert.priority,
                    'Confidence': f"{alert.confidence:.1%}",
                    'Action': alert.action_recommended
                })
            
            df_alerts = pd.DataFrame(alert_data)
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.info("No recent triggered alerts")
        
        # Export alerts functionality
        st.subheader("💾 Export Alert Data")
        
        if st.button("📊 Export Alert History", type="secondary"):
            # Create export data
            all_alerts = alert_system.get_active_alerts() + alert_system.get_triggered_alerts(hours=8760)  # 1 year
            
            if all_alerts:
                export_data = []
                for alert in all_alerts:
                    export_data.append({
                        'Alert_ID': alert.id,
                        'Symbol': alert.symbol,
                        'Alert_Type': alert.alert_type,
                        'Condition': alert.condition,
                        'Target_Value': alert.target_value,
                        'Current_Value': alert.current_value,
                        'Priority': alert.priority,
                        'Status': 'Triggered' if alert.triggered_at else 'Active',
                        'Created_Date': alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'Triggered_Date': alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S') if alert.triggered_at else '',
                        'Confidence': f"{alert.confidence:.1%}",
                        'Recommended_Action': alert.action_recommended,
                        'Message': alert.message
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Alert History (CSV)",
                    data=csv,
                    file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No alert data to export")



def profit_confirmation_interface(data_fetcher, pattern_recognition, profit_analyzer):
    """Dedicated interface for candlestick profit confirmation analysis"""
    st.header("💰 Candlestick Formation Profit Confirmation")
    st.markdown("**Professional analysis for day trading, swing trading, and long-term profit potential based on candlestick formations**")
    
    # Input section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        trading_style = st.selectbox(
            "Trading Style",
            ["day_trading", "swing_trading", "long_term"],
            format_func=lambda x: {
                "day_trading": "📈 Day Trading (Intraday)",
                "swing_trading": "📊 Swing Trading (3-10 days)", 
                "long_term": "📈 Long-term (2-16 weeks)"
            }[x],
            help="Select your preferred trading timeframe"
        )
    
    with col2:
        timeframe = st.selectbox("Chart Timeframe", ["1d", "1h", "15min"], index=0)
        confidence_threshold = st.slider("Min Confidence %", 50, 95, 70)
    
    with col3:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y"],
            help="Historical data period for pattern analysis"
        )
    
    # Initialize dedicated profit confirmation state
    analysis_key = f"profit_confirm_{symbol}_{timeframe}_{trading_style}_{analysis_period}"
    if f'analysis_done_{analysis_key}' not in st.session_state:
        st.session_state[f'analysis_done_{analysis_key}'] = False
    
    if not st.session_state[f'analysis_done_{analysis_key}']:
        if st.button("🚀 Generate Profit Confirmation Analysis", type="primary", use_container_width=True, key=f"confirm_btn_{analysis_key}"):
            with st.spinner("Analyzing candlestick formations for profit potential..."):
                try:
                    # Fetch data and run analysis
                    data = data_fetcher.fetch_data(symbol, timeframe, period=analysis_period)
                    
                    if data is not None and not data.empty:
                        # Detect patterns
                        all_pattern_types = ['Hammer', 'Inverted Hammer', 'Shooting Star', 'Hanging Man', 'Doji', 
                                           'Bullish Engulfing', 'Bearish Engulfing', 'Morning Star', 'Evening Star',
                                           'Three White Soldiers', 'Three Black Crows', 'Piercing Pattern', 'Dark Cloud Cover']
                        patterns = pattern_recognition.analyze_patterns(data, all_pattern_types)
                        
                        if patterns:
                            # Run profit analysis
                            profit_analysis = profit_analyzer.analyze_profit_potential(data, patterns, trading_style)
                            
                            # Store results in session state
                            st.session_state[f'profit_data_{analysis_key}'] = profit_analysis
                            st.session_state[f'patterns_data_{analysis_key}'] = patterns
                            st.session_state[f'analysis_done_{analysis_key}'] = True
                            
                            st.success("✅ Profit confirmation analysis completed!")
                        else:
                            st.warning("No candlestick patterns detected in the specified timeframe. Try adjusting the analysis period or timeframe.")
                    else:
                        st.error(f"Unable to fetch data for {symbol}. Please check the symbol and try again.")
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
    else:
        st.success("✅ Analysis completed!")
        if st.button("🔄 Run New Analysis", key=f"refresh_confirm_{analysis_key}"):
            st.session_state[f'analysis_done_{analysis_key}'] = False
    
    # Display results if analysis is completed
    if st.session_state.get(f'analysis_done_{analysis_key}', False):
        if f'profit_data_{analysis_key}' in st.session_state:
            profit_analysis = st.session_state[f'profit_data_{analysis_key}']
            patterns = st.session_state[f'patterns_data_{analysis_key}']
            
            # Display comprehensive analysis
            display_profit_analysis(profit_analysis, symbol)
                    
            # Additional insights section
            st.markdown("---")
            st.subheader("📊 Pattern Formation Details")
            
            # Show individual pattern analysis
            for analysis in profit_analysis['pattern_analyses']:
                with st.expander(f"{analysis['pattern_type']} - Expected {analysis['expected_profit_pct']:.1f}% profit"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Success Rate", f"{analysis['success_rate']:.1%}")
                        st.metric("Entry Price", f"${analysis['entry_price']:.2f}")
                    
                    with col2:
                        st.metric("Target Price", f"${analysis['target_price']:.2f}")
                        st.metric("Stop Loss", f"${analysis['stop_loss']:.2f}")
                    
                    with col3:
                        st.metric("Risk:Reward", f"{analysis['risk_reward_ratio']:.1f}:1")
                        st.metric("Time to Profit", analysis['time_to_profit'])
                    
                    # Direction and strength
                    direction_color = "green" if analysis['direction'] == 'bullish' else "red"
                    st.markdown(f"**Direction:** <span style='color: {direction_color}'>{analysis['direction'].upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Pattern Strength:** {analysis['pattern_strength']}")
                    st.markdown(f"**Volume Confirmation:** {'✅ Yes' if analysis['volume_confirmation'] else '❌ No'}")

def display_profit_analysis(profit_analysis, symbol):
    """Display comprehensive profit analysis results"""
    try:
        overall = profit_analysis['overall_assessment']
        
        # Main recommendation card
        recommendation_color = {
            'STRONG BUY': 'green',
            'BUY': 'green', 
            'WEAK BUY': 'orange',
            'STRONG SELL': 'red',
            'SELL': 'red',
            'WEAK SELL': 'orange',
            'WAIT': 'gray'
        }.get(overall['recommendation'], 'gray')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {recommendation_color}20, {recommendation_color}10); 
                    border: 2px solid {recommendation_color}; 
                    border-radius: 15px; 
                    padding: 25px; 
                    text-align: center; 
                    margin: 20px 0;">
            <h2 style="color: {recommendation_color}; margin: 0;">
                {overall['recommendation']}
            </h2>
            <h3 style="color: {recommendation_color}; margin: 10px 0;">
                Grade: {overall['grade']} | Confidence: {overall['confidence']:.1%}
            </h3>
            <p style="font-size: 18px; margin: 10px 0;">
                Expected Profit: <strong>{overall['expected_profit_pct']:.1f}%</strong> | 
                Risk:Reward: <strong>{overall['risk_reward_ratio']:.1f}:1</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pattern Count", overall['pattern_count'])
        with col2:
            st.metric("Bullish Patterns", overall['bullish_patterns'], 
                     delta=f"+{overall['bullish_patterns'] - overall['bearish_patterns']}")
        with col3:
            st.metric("Bearish Patterns", overall['bearish_patterns'])
        with col4:
            st.metric("Current Price", f"${profit_analysis['current_price']:.2f}")
        
        # Profit targets
        if profit_analysis['profit_targets']:
            st.subheader("🎯 Profit Targets")
            target_col1, target_col2, target_col3 = st.columns(3)
            
            targets = profit_analysis['profit_targets']
            with target_col1:
                if 'conservative' in targets:
                    st.metric("Conservative Target", f"${targets['conservative']:.2f}", 
                             f"{((targets['conservative'] / profit_analysis['current_price']) - 1) * 100:+.1f}%")
            
            with target_col2:
                if 'realistic' in targets:
                    st.metric("Realistic Target", f"${targets['realistic']:.2f}",
                             f"{((targets['realistic'] / profit_analysis['current_price']) - 1) * 100:+.1f}%")
            
            with target_col3:
                if 'optimistic' in targets:
                    st.metric("Optimistic Target", f"${targets['optimistic']:.2f}",
                             f"{((targets['optimistic'] / profit_analysis['current_price']) - 1) * 100:+.1f}%")
        
        # Risk factors and confirmations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Risk Factors")
            risk_factors = profit_analysis['risk_factors']
            if risk_factors:
                for risk in risk_factors:
                    st.warning(f"• {risk}")
            else:
                st.success("• No major risk factors identified")
        
        with col2:
            st.subheader("✅ Confirmation Signals")
            confirmations = profit_analysis['confirmation_signals']
            
            if 'volume_trend' in confirmations:
                trend_color = {"increasing": "green", "decreasing": "red", "neutral": "orange"}[confirmations['volume_trend']]
                st.markdown(f"• **Volume Trend:** <span style='color: {trend_color}'>{confirmations['volume_trend']}</span>", unsafe_allow_html=True)
            
            if 'price_momentum' in confirmations:
                momentum_color = {"bullish": "green", "bearish": "red", "neutral": "orange"}[confirmations['price_momentum']]
                st.markdown(f"• **Price Momentum:** <span style='color: {momentum_color}'>{confirmations['price_momentum']}</span>", unsafe_allow_html=True)
            
            if 'technical_indicators' in confirmations and confirmations['technical_indicators']:
                for indicator in confirmations['technical_indicators']:
                    st.success(f"• {indicator}")
        
        # Entry/Exit strategy
        if profit_analysis['entry_exit_strategy']:
            st.subheader("📋 Entry/Exit Strategy")
            strategy = profit_analysis['entry_exit_strategy']
            
            strategy_col1, strategy_col2 = st.columns(2)
            
            with strategy_col1:
                st.markdown(f"**Entry Timing:** {strategy.get('entry_timing', 'N/A').replace('_', ' ').title()}")
                st.markdown(f"**Position Sizing:** {strategy.get('position_sizing', 'N/A').title()}")
            
            with strategy_col2:
                st.markdown(f"**Exit Strategy:** {strategy.get('exit_strategy', 'N/A').replace('_', ' ').title()}")
                st.markdown(f"**Timeframe:** {strategy.get('timeframe', 'N/A').replace('_', ' ').title()}")
        
        # Trading style info
        style_info = {
            'day_trading': "⚡ **Day Trading:** Focus on intraday price movements with quick entries and exits. Higher success rates but smaller profits per trade.",
            'swing_trading': "📊 **Swing Trading:** Hold positions for several days to capture medium-term price swings. Balanced risk-reward approach.",
            'long_term': "📈 **Long-term:** Position trading over weeks to months. Lower frequency but potentially higher profits per trade."
        }
        
        st.info(style_info.get(profit_analysis['trading_style'], "Trading style information not available."))
        
    except Exception as e:
        st.error(f"Error displaying profit analysis: {str(e)}")

if __name__ == "__main__":
    main()
