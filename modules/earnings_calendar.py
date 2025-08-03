"""
Earnings Calendar Module
Track upcoming earnings and analyze earnings impact on stock prices
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class EarningsEvent:
    """Individual earnings event data"""
    symbol: str
    company_name: str
    earnings_date: datetime
    estimated_eps: Optional[float]
    actual_eps: Optional[float]
    surprise_pct: Optional[float]
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    market_cap: Optional[float]
    sector: str
    pre_market: bool  # True if before market open
    confirmed: bool  # True if date is confirmed

@dataclass
class EarningsAnalysis:
    """Earnings analysis results"""
    symbol: str
    current_price: float
    earnings_date: datetime
    days_until_earnings: int
    historical_moves: Dict[str, float]  # avg move, up_pct, down_pct
    iv_percentile: Optional[float]
    analyst_expectations: Dict[str, float]
    risk_assessment: str  # LOW, MEDIUM, HIGH
    trading_strategy: str
    key_metrics: Dict[str, float]

class EarningsCalendar:
    """Earnings calendar and analysis system"""
    
    def __init__(self):
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        
    def get_upcoming_earnings(self, days_ahead: int = 60) -> List[EarningsEvent]:
        """Get upcoming earnings events"""
        try:
            # In production, this would connect to earnings calendar APIs
            # For demo, we'll create realistic upcoming earnings for full 60-day period
            
            upcoming_events = []
            base_date = datetime.now()
            
            # Create demo earnings events for the next 60 days
            demo_earnings = [
                # Week 1 (1-7 days)
                ('JPM', 'JPMorgan Chase', 3, 3.15, None, 'Finance', True),
                ('JNJ', 'Johnson & Johnson', 5, 2.42, None, 'Healthcare', False),
                ('AAPL', 'Apple Inc.', 7, 1.52, None, 'Technology', False),
                
                # Week 2 (8-14 days)
                ('MSFT', 'Microsoft Corp.', 10, 2.78, None, 'Technology', False),
                ('BAC', 'Bank of America', 12, 0.85, None, 'Finance', True),
                ('GOOGL', 'Alphabet Inc.', 14, 1.45, None, 'Technology', True),
                
                # Week 3 (15-21 days)
                ('PFE', 'Pfizer Inc.', 16, 1.35, None, 'Healthcare', False),
                ('AMZN', 'Amazon.com Inc.', 18, 0.75, None, 'Consumer', True),
                ('WFC', 'Wells Fargo', 19, 1.25, None, 'Finance', True),
                ('TSLA', 'Tesla Inc.', 21, 0.85, None, 'Consumer', True),
                
                # Week 4 (22-28 days)
                ('GS', 'Goldman Sachs', 23, 8.45, None, 'Finance', False),
                ('NVDA', 'NVIDIA Corp.', 25, 3.25, None, 'Technology', False),
                ('UNH', 'UnitedHealth Group', 26, 5.75, None, 'Healthcare', False),
                ('HD', 'The Home Depot', 28, 3.85, None, 'Consumer', False),
                
                # Week 5 (29-35 days)
                ('META', 'Meta Platforms', 30, 2.95, None, 'Technology', True),
                ('XOM', 'Exxon Mobil', 32, 1.95, None, 'Energy', False),
                ('ABBV', 'AbbVie Inc.', 33, 2.85, None, 'Healthcare', False),
                ('NKE', 'Nike Inc.', 35, 0.95, None, 'Consumer', True),
                
                # Week 6 (36-42 days)
                ('CVX', 'Chevron Corp.', 37, 2.45, None, 'Energy', False),
                ('MS', 'Morgan Stanley', 39, 1.85, None, 'Finance', True),
                ('TMO', 'Thermo Fisher Scientific', 40, 4.25, None, 'Healthcare', False),
                ('SBUX', 'Starbucks Corp.', 42, 0.75, None, 'Consumer', False),
                
                # Week 7 (43-49 days)
                ('COP', 'ConocoPhillips', 44, 1.75, None, 'Energy', True),
                ('IBM', 'International Business Machines', 46, 1.95, None, 'Technology', False),
                ('KO', 'The Coca-Cola Company', 47, 0.65, None, 'Consumer', False),
                ('V', 'Visa Inc.', 49, 1.85, None, 'Finance', False),
                
                # Week 8 (50-56 days)
                ('EOG', 'EOG Resources', 51, 1.35, None, 'Energy', True),
                ('INTC', 'Intel Corp.', 53, 0.45, None, 'Technology', True),
                ('PG', 'Procter & Gamble', 54, 1.45, None, 'Consumer', False),
                ('SLB', 'Schlumberger', 56, 0.85, None, 'Energy', False),
                
                # Week 9 (57-60 days)
                ('ORCL', 'Oracle Corp.', 58, 1.25, None, 'Technology', False),
                ('MCD', 'McDonald\'s Corp.', 60, 2.85, None, 'Consumer', False)
            ]
            
            for symbol, name, days_out, eps_est, eps_actual, sector, pre_market in demo_earnings:
                if days_out <= days_ahead:
                    earnings_date = base_date + timedelta(days=days_out)
                    
                    # Get market cap from yfinance
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        market_cap = info.get('marketCap', 0)
                    except:
                        market_cap = 0
                    
                    event = EarningsEvent(
                        symbol=symbol,
                        company_name=name,
                        earnings_date=earnings_date,
                        estimated_eps=eps_est,
                        actual_eps=eps_actual,
                        surprise_pct=None,
                        revenue_estimate=market_cap * 0.0001 if market_cap > 0 else None,  # Demo calc
                        revenue_actual=None,
                        market_cap=market_cap,
                        sector=sector,
                        pre_market=pre_market,
                        confirmed=True
                    )
                    
                    upcoming_events.append(event)
            
            return sorted(upcoming_events, key=lambda x: x.earnings_date)
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    def analyze_earnings_impact(self, symbol: str) -> Optional[EarningsAnalysis]:
        """Analyze earnings impact for a specific symbol"""
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y')
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Find next earnings date (demo)
            upcoming_earnings = self.get_upcoming_earnings()
            earnings_event = next((e for e in upcoming_earnings if e.symbol == symbol), None)
            
            if not earnings_event:
                # Create a demo earnings event
                earnings_date = datetime.now() + timedelta(days=15)
                days_until = 15
            else:
                earnings_date = earnings_event.earnings_date
                days_until = max(0, (earnings_date - datetime.now()).days)
            
            # Calculate historical earnings moves (demo)
            historical_moves = self._calculate_historical_moves(symbol, hist)
            
            # Get options IV percentile (demo)
            iv_percentile = self._get_iv_percentile(symbol)
            
            # Analyst expectations (demo)
            analyst_expectations = {
                'price_target': current_price * 1.05,
                'eps_growth': 12.5,
                'revenue_growth': 8.3,
                'buy_ratings': 15,
                'hold_ratings': 8,
                'sell_ratings': 2
            }
            
            # Risk assessment
            risk_assessment = self._assess_earnings_risk(days_until, historical_moves, iv_percentile)
            
            # Trading strategy
            trading_strategy = self._generate_trading_strategy(risk_assessment, historical_moves, days_until)
            
            # Key metrics
            key_metrics = {
                'avg_move': historical_moves['avg_move'],
                'upside_probability': historical_moves['up_pct'],
                'iv_rank': iv_percentile or 50,
                'analyst_score': (analyst_expectations['buy_ratings'] * 2 + analyst_expectations['hold_ratings']) / 
                               (analyst_expectations['buy_ratings'] + analyst_expectations['hold_ratings'] + analyst_expectations['sell_ratings']) * 50
            }
            
            return EarningsAnalysis(
                symbol=symbol,
                current_price=current_price,
                earnings_date=earnings_date,
                days_until_earnings=days_until,
                historical_moves=historical_moves,
                iv_percentile=iv_percentile,
                analyst_expectations=analyst_expectations,
                risk_assessment=risk_assessment,
                trading_strategy=trading_strategy,
                key_metrics=key_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing earnings impact for {symbol}: {e}")
            return None
    
    def _calculate_historical_moves(self, symbol: str, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate historical earnings moves (demo implementation)"""
        # In production, this would analyze actual earnings announcement dates
        # For demo, we'll simulate based on historical volatility
        
        daily_returns = hist['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
        
        # Estimate earnings moves based on volatility
        avg_earnings_move = volatility * 0.15  # Typical earnings move is ~15% of annual vol
        
        # Simulate historical win rate
        up_percentage = 55 if symbol in ['AAPL', 'MSFT', 'GOOGL'] else 45  # Growth stocks tend to beat more
        
        return {
            'avg_move': avg_earnings_move * 100,  # Convert to percentage
            'up_pct': up_percentage,
            'down_pct': 100 - up_percentage,
            'max_move': avg_earnings_move * 2.5 * 100,
            'min_move': avg_earnings_move * 0.3 * 100
        }
    
    def _get_iv_percentile(self, symbol: str) -> Optional[float]:
        """Get implied volatility percentile (demo)"""
        # In production, this would fetch real options data
        # For demo, return reasonable IV percentiles
        
        high_iv_stocks = ['TSLA', 'NVDA', 'AMD', 'NFLX']
        medium_iv_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        if symbol in high_iv_stocks:
            return 75.0
        elif symbol in medium_iv_stocks:
            return 55.0
        else:
            return 45.0
    
    def _assess_earnings_risk(self, days_until: int, historical_moves: Dict, iv_percentile: Optional[float]) -> str:
        """Assess earnings risk level"""
        risk_score = 0
        
        # Time until earnings
        if days_until <= 3:
            risk_score += 3
        elif days_until <= 7:
            risk_score += 2
        else:
            risk_score += 1
        
        # Historical volatility
        if historical_moves['avg_move'] > 10:
            risk_score += 3
        elif historical_moves['avg_move'] > 5:
            risk_score += 2
        else:
            risk_score += 1
        
        # IV percentile
        if iv_percentile and iv_percentile > 70:
            risk_score += 2
        elif iv_percentile and iv_percentile > 50:
            risk_score += 1
        
        if risk_score >= 7:
            return 'HIGH'
        elif risk_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_trading_strategy(self, risk_assessment: str, historical_moves: Dict, days_until: int) -> str:
        """Generate earnings trading strategy"""
        if risk_assessment == 'HIGH':
            if days_until <= 3:
                return 'AVOID_NEW_POSITIONS'
            else:
                return 'SELL_OPTIONS'
        elif risk_assessment == 'MEDIUM':
            if historical_moves['up_pct'] > 60:
                return 'BULLISH_SPREAD'
            else:
                return 'NEUTRAL_STRATEGY'
        else:
            return 'BUY_AND_HOLD'

def earnings_calendar_interface():
    """Streamlit interface for earnings calendar"""
    st.header("📅 Earnings Calendar & Analysis")
    st.markdown("**Track upcoming earnings and analyze their potential market impact**")
    
    # Initialize calendar
    if 'earnings_calendar' not in st.session_state:
        st.session_state.earnings_calendar = EarningsCalendar()
    
    calendar = st.session_state.earnings_calendar
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📅 Upcoming Earnings", "🔍 Individual Analysis", "📊 Sector Overview"])
    
    with tab1:
        st.subheader("Upcoming Earnings Events")
        
        days_filter = st.selectbox("Show earnings for:", [7, 14, 30, 60], index=1, format_func=lambda x: f"Next {x} days")
        
        with st.spinner("Loading earnings calendar..."):
            upcoming_earnings = calendar.get_upcoming_earnings(days_filter)
        
        if upcoming_earnings:
            # Create earnings table
            earnings_data = []
            for event in upcoming_earnings:
                days_until = max(0, (event.earnings_date - datetime.now()).days)
                
                earnings_data.append({
                    'Symbol': event.symbol,
                    'Company': event.company_name,
                    'Date': event.earnings_date.strftime('%Y-%m-%d'),
                    'Days Until': days_until,
                    'Time': 'Pre-Market' if event.pre_market else 'After Close',
                    'Sector': event.sector,
                    'Est. EPS': f"${event.estimated_eps:.2f}" if event.estimated_eps else "N/A",
                    'Market Cap': f"${event.market_cap/1e9:.1f}B" if event.market_cap else "N/A"
                })
            
            df_earnings = pd.DataFrame(earnings_data)
            st.dataframe(df_earnings, use_container_width=True)
            
            # Highlight this week's earnings
            this_week = [e for e in upcoming_earnings if (e.earnings_date - datetime.now()).days <= 7]
            if this_week:
                st.subheader("🔥 This Week's Key Earnings")
                for event in this_week:
                    with st.expander(f"{event.symbol} - {event.company_name} ({event.earnings_date.strftime('%m/%d')})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Sector:** {event.sector}")
                            st.write(f"**Expected EPS:** ${event.estimated_eps:.2f}" if event.estimated_eps else "**Expected EPS:** N/A")
                            st.write(f"**Timing:** {'Pre-Market' if event.pre_market else 'After Close'}")
                        with col2:
                            st.write(f"**Market Cap:** ${event.market_cap/1e9:.1f}B" if event.market_cap else "**Market Cap:** N/A")
                            st.write(f"**Date Confirmed:** {'Yes' if event.confirmed else 'Estimated'}")
                            if event.symbol:
                                st.write(f"[Analyze {event.symbol}](#) ← Click Individual Analysis tab")
        else:
            st.info("No earnings events found for the selected period.")
    
    with tab2:
        st.subheader("Individual Stock Earnings Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL").upper()
        with col2:
            if st.button("📊 Analyze Earnings Impact", type="primary"):
                if symbol:
                    with st.spinner(f"Analyzing earnings impact for {symbol}..."):
                        analysis = calendar.analyze_earnings_impact(symbol)
                    
                    if analysis:
                        # Key metrics dashboard
                        st.subheader(f"📈 {symbol} Earnings Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${analysis.current_price:.2f}")
                        with col2:
                            st.metric("Days Until Earnings", analysis.days_until_earnings)
                        with col3:
                            risk_color = "red" if analysis.risk_assessment == "HIGH" else "orange" if analysis.risk_assessment == "MEDIUM" else "green"
                            st.markdown(f"<div style='color: {risk_color}'>**Risk Level:** {analysis.risk_assessment}</div>", unsafe_allow_html=True)
                        with col4:
                            st.metric("Avg Historical Move", f"{analysis.historical_moves['avg_move']:.1f}%")
                        
                        # Historical performance
                        st.subheader("📊 Historical Earnings Performance")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Historical Moves:**")
                            st.write(f"• Average Move: ±{analysis.historical_moves['avg_move']:.1f}%")
                            st.write(f"• Maximum Move: {analysis.historical_moves['max_move']:.1f}%")
                            st.write(f"• Minimum Move: {analysis.historical_moves['min_move']:.1f}%")
                            
                            st.markdown("**Direction Probability:**")
                            st.write(f"• Up: {analysis.historical_moves['up_pct']:.0f}%")
                            st.write(f"• Down: {analysis.historical_moves['down_pct']:.0f}%")
                        
                        with col2:
                            st.markdown("**Analyst Expectations:**")
                            st.write(f"• Price Target: ${analysis.analyst_expectations['price_target']:.2f}")
                            st.write(f"• EPS Growth: {analysis.analyst_expectations['eps_growth']:.1f}%")
                            st.write(f"• Revenue Growth: {analysis.analyst_expectations['revenue_growth']:.1f}%")
                            
                            st.markdown("**Analyst Ratings:**")
                            total_ratings = sum([analysis.analyst_expectations['buy_ratings'], analysis.analyst_expectations['hold_ratings'], analysis.analyst_expectations['sell_ratings']])
                            st.write(f"• Buy: {analysis.analyst_expectations['buy_ratings']} ({analysis.analyst_expectations['buy_ratings']/total_ratings*100:.0f}%)")
                            st.write(f"• Hold: {analysis.analyst_expectations['hold_ratings']} ({analysis.analyst_expectations['hold_ratings']/total_ratings*100:.0f}%)")
                            st.write(f"• Sell: {analysis.analyst_expectations['sell_ratings']} ({analysis.analyst_expectations['sell_ratings']/total_ratings*100:.0f}%)")
                        
                        # Options analysis
                        if analysis.iv_percentile:
                            st.subheader("📈 Options Analysis")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                iv_color = "red" if analysis.iv_percentile > 70 else "orange" if analysis.iv_percentile > 50 else "green"
                                st.markdown(f"<div style='color: {iv_color}'>**IV Percentile:** {analysis.iv_percentile:.0f}%</div>", unsafe_allow_html=True)
                                
                                if analysis.iv_percentile > 70:
                                    st.write("📈 **High IV** - Consider selling options")
                                elif analysis.iv_percentile > 50:
                                    st.write("📊 **Moderate IV** - Neutral options strategy")
                                else:
                                    st.write("📉 **Low IV** - Consider buying options")
                            
                            with col2:
                                st.markdown("**Options Strategies:**")
                                if analysis.trading_strategy == 'SELL_OPTIONS':
                                    st.write("• Sell covered calls")
                                    st.write("• Sell cash-secured puts")
                                    st.write("• Iron condors")
                                elif analysis.trading_strategy == 'BULLISH_SPREAD':
                                    st.write("• Bull call spreads")
                                    st.write("• Long calls (limited risk)")
                                else:
                                    st.write("• Buy protective puts")
                                    st.write("• Long stock position")
                        
                        # Trading strategy
                        st.subheader("💡 Recommended Strategy")
                        
                        strategy_recommendations = {
                            'AVOID_NEW_POSITIONS': "⚠️ **High Risk** - Avoid new positions. Earnings too close with high volatility.",
                            'SELL_OPTIONS': "📈 **Income Strategy** - Sell options to collect premium from high IV.",
                            'BULLISH_SPREAD': "🎯 **Bullish Play** - Use defined risk strategies for upside exposure.",
                            'NEUTRAL_STRATEGY': "⚖️ **Stay Neutral** - Mixed signals suggest avoiding directional bets.",
                            'BUY_AND_HOLD': "📊 **Low Risk** - Suitable for long-term positions with limited earnings volatility."
                        }
                        
                        recommendation = strategy_recommendations.get(analysis.trading_strategy, "Monitor for opportunities")
                        st.markdown(recommendation)
                        
                        # Risk warnings
                        st.warning(f"""
                        **Risk Considerations:**
                        • Earnings can cause significant price volatility ({analysis.historical_moves['avg_move']:.1f}% average move)
                        • Options may experience IV crush after earnings
                        • Consider position sizing and stop losses
                        • Monitor pre-market trading on earnings day
                        """)
                        
                    else:
                        st.error(f"Could not analyze earnings impact for {symbol}. Please check the symbol and try again.")
                else:
                    st.error("Please enter a valid stock symbol")
    
    with tab3:
        st.subheader("📊 Sector Earnings Overview")
        
        # Show earnings by sector
        upcoming_earnings = calendar.get_upcoming_earnings(60)
        
        if upcoming_earnings:
            # Group by sector
            sector_earnings = {}
            for event in upcoming_earnings:
                if event.sector not in sector_earnings:
                    sector_earnings[event.sector] = []
                sector_earnings[event.sector].append(event)
            
            for sector, events in sector_earnings.items():
                with st.expander(f"{sector} Sector ({len(events)} companies)"):
                    for event in events:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**{event.symbol}** - {event.company_name}")
                        with col2:
                            st.write(f"📅 {event.earnings_date.strftime('%m/%d/%Y')}")
                        with col3:
                            st.write(f"💰 Est. EPS: ${event.estimated_eps:.2f}" if event.estimated_eps else "💰 Est. EPS: N/A")
        
        # Sector rotation insights
        st.subheader("🔄 Sector Rotation Insights")
        st.info("""
        **Key Sector Themes for Earnings Season:**
        
        📱 **Technology**: Focus on AI growth, cloud revenue, and margin expansion
        
        🏥 **Healthcare**: Drug approvals, pipeline updates, and regulatory news
        
        🏦 **Finance**: Interest rate impacts, loan growth, and credit quality
        
        🛒 **Consumer**: Spending patterns, inflation impacts, and supply chain
        
        ⚡ **Energy**: Oil prices, capital allocation, and renewable transition
        """)
    
    # Help section
    st.markdown("---")
    st.subheader("📚 How to Use Earnings Analysis")
    
    with st.expander("📖 Complete Earnings Trading Guide"):
        st.markdown("""
        **Before Earnings (1-2 weeks out):**
        - Monitor IV percentile for options opportunities
        - Review analyst expectations and revisions
        - Check historical earnings performance
        - Set up price alerts for key levels
        
        **Week of Earnings:**
        - Reduce position sizes if high volatility expected
        - Consider protective strategies (stops, puts)
        - Monitor pre-earnings guidance or leaks
        - Prepare for gap moves
        
        **Day of Earnings:**
        - Avoid new positions close to announcement
        - Monitor pre-market/after-hours trading
        - Be ready to react to earnings surprise
        - Watch for unusual options activity
        
        **After Earnings:**
        - Assess actual vs expected results
        - Monitor guidance changes
        - Look for follow-through price action
        - Consider IV crush impact on options
        
        **Key Metrics to Watch:**
        - **EPS Surprise**: Actual vs expected earnings per share
        - **Revenue Growth**: Quarter-over-quarter and year-over-year
        - **Guidance**: Forward-looking company projections
        - **Margin Trends**: Profitability improvements or declines
        - **Segment Performance**: Business unit specific results
        """)
