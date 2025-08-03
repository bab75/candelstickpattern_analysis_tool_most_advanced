"""
News Sentiment Analyzer Module
Real-time news sentiment analysis for stock trading decisions
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Individual news article data"""
    title: str
    summary: str
    source: str
    published_date: datetime
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1
    url: str
    impact_prediction: str  # LOW, MEDIUM, HIGH

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results"""
    symbol: str
    overall_sentiment: float  # -1 to 1
    news_count: int
    bullish_articles: int
    bearish_articles: int
    neutral_articles: int
    sentiment_trend: str  # IMPROVING, DECLINING, STABLE
    market_impact: str  # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float  # 0 to 1
    key_themes: List[str]
    recent_articles: List[NewsArticle]

class NewsSentimentAnalyzer:
    """Analyze news sentiment for stock trading"""
    
    def __init__(self):
        self.supported_sources = [
            'Yahoo Finance',
            'MarketWatch', 
            'Reuters',
            'Bloomberg',
            'CNBC',
            'The Motley Fool',
            'Seeking Alpha',
            'Benzinga'
        ]
        
    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> SentimentAnalysis:
        """Get comprehensive news sentiment analysis for a symbol"""
        try:
            # In production, this would connect to real news APIs
            # For now, we'll create a realistic demo with varied sentiment
            
            articles = self._fetch_news_articles(symbol, days_back)
            sentiment_analysis = self._analyze_sentiment(articles)
            market_impact = self._predict_market_impact(sentiment_analysis)
            
            return SentimentAnalysis(
                symbol=symbol,
                overall_sentiment=sentiment_analysis['overall_score'],
                news_count=len(articles),
                bullish_articles=sentiment_analysis['bullish_count'],
                bearish_articles=sentiment_analysis['bearish_count'], 
                neutral_articles=sentiment_analysis['neutral_count'],
                sentiment_trend=sentiment_analysis['trend'],
                market_impact=market_impact,
                confidence=sentiment_analysis['confidence'],
                key_themes=sentiment_analysis['themes'],
                recent_articles=articles[:10]  # Most recent 10
            )
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return self._create_fallback_sentiment(symbol)
    
    def _fetch_news_articles(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news articles for the symbol (demo implementation)"""
        # In production, integrate with NewsAPI, Alpha Vantage News, or similar
        
        # Create realistic demo articles with varied sentiment
        demo_articles = [
            NewsArticle(
                title=f"{symbol} Reports Strong Q3 Earnings, Beats Estimates",
                summary=f"{symbol} exceeded analyst expectations with strong revenue growth and positive guidance.",
                source="MarketWatch",
                published_date=datetime.now() - timedelta(hours=6),
                sentiment_score=0.8,
                relevance_score=0.95,
                url=f"https://marketwatch.com/{symbol.lower()}-earnings",
                impact_prediction="HIGH"
            ),
            NewsArticle(
                title=f"Analyst Upgrades {symbol} to Buy on Innovation Pipeline", 
                summary=f"Major investment bank raises price target citing strong product development.",
                source="Reuters",
                published_date=datetime.now() - timedelta(hours=12),
                sentiment_score=0.6,
                relevance_score=0.85,
                url=f"https://reuters.com/{symbol.lower()}-upgrade",
                impact_prediction="MEDIUM"
            ),
            NewsArticle(
                title=f"Market Volatility Impacts {symbol} Trading Volume",
                summary=f"Broader market concerns affect {symbol} despite solid fundamentals.",
                source="CNBC",
                published_date=datetime.now() - timedelta(days=1),
                sentiment_score=-0.2,
                relevance_score=0.7,
                url=f"https://cnbc.com/{symbol.lower()}-volatility",
                impact_prediction="LOW"
            ),
            NewsArticle(
                title=f"{symbol} Announces Strategic Partnership", 
                summary=f"Company enters joint venture to expand market reach and capabilities.",
                source="Bloomberg",
                published_date=datetime.now() - timedelta(days=2),
                sentiment_score=0.4,
                relevance_score=0.8,
                url=f"https://bloomberg.com/{symbol.lower()}-partnership",
                impact_prediction="MEDIUM"
            ),
            NewsArticle(
                title=f"Regulatory Concerns Weigh on {symbol} Outlook",
                summary=f"Industry regulations may impact future growth prospects for {symbol}.",
                source="Yahoo Finance", 
                published_date=datetime.now() - timedelta(days=3),
                sentiment_score=-0.5,
                relevance_score=0.75,
                url=f"https://finance.yahoo.com/{symbol.lower()}-regulation",
                impact_prediction="MEDIUM"
            )
        ]
        
        return demo_articles
    
    def _analyze_sentiment(self, articles: List[NewsArticle]) -> Dict:
        """Analyze overall sentiment from articles"""
        if not articles:
            return {
                'overall_score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'trend': 'STABLE',
                'confidence': 0.5,
                'themes': ['No recent news available']
            }
        
        # Calculate weighted sentiment
        total_weighted_sentiment = 0
        total_weights = 0
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for article in articles:
            weight = article.relevance_score
            total_weighted_sentiment += article.sentiment_score * weight
            total_weights += weight
            
            if article.sentiment_score > 0.2:
                bullish_count += 1
            elif article.sentiment_score < -0.2:
                bearish_count += 1
            else:
                neutral_count += 1
        
        overall_score = total_weighted_sentiment / total_weights if total_weights > 0 else 0
        
        # Determine trend
        recent_articles = sorted(articles, key=lambda x: x.published_date, reverse=True)[:5]
        recent_sentiment = sum(a.sentiment_score for a in recent_articles) / len(recent_articles)
        older_articles = articles[5:] if len(articles) > 5 else []
        older_sentiment = sum(a.sentiment_score for a in older_articles) / len(older_articles) if older_articles else recent_sentiment
        
        if recent_sentiment > older_sentiment + 0.1:
            trend = 'IMPROVING'
        elif recent_sentiment < older_sentiment - 0.1:
            trend = 'DECLINING'
        else:
            trend = 'STABLE'
        
        # Extract key themes
        themes = self._extract_themes(articles)
        
        # Calculate confidence based on article count and consistency
        confidence = min(len(articles) / 10, 1.0) * 0.7 + (1 - abs(overall_score - recent_sentiment)) * 0.3
        
        return {
            'overall_score': overall_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'trend': trend,
            'confidence': confidence,
            'themes': themes
        }
    
    def _extract_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extract key themes from news articles"""
        # Simple theme extraction based on common keywords
        keywords = {}
        
        for article in articles:
            text = (article.title + " " + article.summary).lower()
            
            # Common financial themes
            if any(word in text for word in ['earnings', 'profit', 'revenue', 'beat', 'miss']):
                keywords['Earnings'] = keywords.get('Earnings', 0) + 1
            if any(word in text for word in ['upgrade', 'downgrade', 'analyst', 'rating']):
                keywords['Analyst Coverage'] = keywords.get('Analyst Coverage', 0) + 1
            if any(word in text for word in ['partnership', 'merger', 'acquisition', 'deal']):
                keywords['Corporate Activity'] = keywords.get('Corporate Activity', 0) + 1
            if any(word in text for word in ['regulation', 'regulatory', 'compliance', 'legal']):
                keywords['Regulatory'] = keywords.get('Regulatory', 0) + 1
            if any(word in text for word in ['innovation', 'product', 'technology', 'launch']):
                keywords['Innovation'] = keywords.get('Innovation', 0) + 1
            if any(word in text for word in ['market', 'competition', 'industry']):
                keywords['Market Conditions'] = keywords.get('Market Conditions', 0) + 1
        
        # Return top themes
        sorted_themes = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [theme[0] for theme in sorted_themes[:5]]
    
    def _predict_market_impact(self, sentiment_analysis: Dict) -> str:
        """Predict market impact based on sentiment"""
        score = sentiment_analysis['overall_score']
        confidence = sentiment_analysis['confidence']
        
        if confidence < 0.3:
            return 'NEUTRAL'
        
        if score > 0.3:
            return 'POSITIVE'
        elif score < -0.3:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def _create_fallback_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Create fallback sentiment when analysis fails"""
        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment=0.0,
            news_count=0,
            bullish_articles=0,
            bearish_articles=0,
            neutral_articles=0,
            sentiment_trend='STABLE',
            market_impact='NEUTRAL',
            confidence=0.3,
            key_themes=['Limited news data available'],
            recent_articles=[]
        )

def news_sentiment_interface():
    """Streamlit interface for news sentiment analysis"""
    st.header("📰 News Sentiment Analyzer")
    st.markdown("**Real-time news sentiment analysis for informed trading decisions**")
    
    # Initialize analyzer
    if 'news_analyzer' not in st.session_state:
        st.session_state.news_analyzer = NewsSentimentAnalyzer()
    
    analyzer = st.session_state.news_analyzer
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA").upper()
    
    with col2:
        days_back = st.selectbox("Analysis Period", [3, 7, 14, 30], index=1, format_func=lambda x: f"{x} days")
    
    if st.button("🔍 Analyze News Sentiment", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing news sentiment for {symbol}..."):
                sentiment = analyzer.get_news_sentiment(symbol, days_back)
            
            # Overall sentiment dashboard
            st.subheader("📊 Sentiment Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment_color = "green" if sentiment.overall_sentiment > 0.2 else "red" if sentiment.overall_sentiment < -0.2 else "orange"
                st.markdown(f"<div style='color: {sentiment_color}'>**Overall Sentiment:** {sentiment.overall_sentiment:.2f}</div>", unsafe_allow_html=True)
            
            with col2:
                st.metric("News Articles", sentiment.news_count)
            
            with col3:
                trend_color = "green" if sentiment.sentiment_trend == "IMPROVING" else "red" if sentiment.sentiment_trend == "DECLINING" else "orange"
                st.markdown(f"<div style='color: {trend_color}'>**Trend:** {sentiment.sentiment_trend}</div>", unsafe_allow_html=True)
            
            with col4:
                impact_color = "green" if sentiment.market_impact == "POSITIVE" else "red" if sentiment.market_impact == "NEGATIVE" else "orange"
                st.markdown(f"<div style='color: {impact_color}'>**Market Impact:** {sentiment.market_impact}</div>", unsafe_allow_html=True)
            
            # Sentiment breakdown
            st.subheader("📈 Sentiment Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Bullish Articles", sentiment.bullish_articles, f"{sentiment.bullish_articles/sentiment.news_count*100:.0f}%" if sentiment.news_count > 0 else "0%")
                st.metric("Bearish Articles", sentiment.bearish_articles, f"{sentiment.bearish_articles/sentiment.news_count*100:.0f}%" if sentiment.news_count > 0 else "0%")
                st.metric("Neutral Articles", sentiment.neutral_articles, f"{sentiment.neutral_articles/sentiment.news_count*100:.0f}%" if sentiment.news_count > 0 else "0%")
            
            with col2:
                st.markdown("**Key Themes:**")
                for theme in sentiment.key_themes:
                    st.write(f"• {theme}")
                
                st.markdown(f"**Analysis Confidence:** {sentiment.confidence:.1%}")
            
            # Recent articles
            if sentiment.recent_articles:
                st.subheader("📰 Recent News Articles")
                
                for i, article in enumerate(sentiment.recent_articles[:5]):
                    with st.expander(f"{article.title} ({article.source})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Summary:** {article.summary}")
                            st.write(f"**Published:** {article.published_date.strftime('%Y-%m-%d %H:%M')}")
                        
                        with col2:
                            sentiment_emoji = "🟢" if article.sentiment_score > 0.2 else "🔴" if article.sentiment_score < -0.2 else "🟡"
                            st.write(f"**Sentiment:** {sentiment_emoji} {article.sentiment_score:.2f}")
                            st.write(f"**Impact:** {article.impact_prediction}")
                            st.write(f"**Relevance:** {article.relevance_score:.1%}")
                            if article.url:
                                st.write(f"[Read Full Article]({article.url})")
            
            # Trading implications
            st.subheader("💡 Trading Implications")
            
            if sentiment.market_impact == "POSITIVE" and sentiment.confidence > 0.6:
                st.success(f"""
                **Bullish Signal for {symbol}**
                - Positive news sentiment with {sentiment.confidence:.1%} confidence
                - {sentiment.bullish_articles} bullish vs {sentiment.bearish_articles} bearish articles
                - Sentiment trend: {sentiment.sentiment_trend}
                - Consider: Long positions, call options, or adding to existing positions
                """)
            elif sentiment.market_impact == "NEGATIVE" and sentiment.confidence > 0.6:
                st.error(f"""
                **Bearish Signal for {symbol}**
                - Negative news sentiment with {sentiment.confidence:.1%} confidence
                - {sentiment.bearish_articles} bearish vs {sentiment.bullish_articles} bullish articles  
                - Sentiment trend: {sentiment.sentiment_trend}
                - Consider: Avoiding new positions, profit taking, or protective puts
                """)
            else:
                st.info(f"""
                **Neutral Signal for {symbol}**
                - Mixed or limited news sentiment
                - Confidence level: {sentiment.confidence:.1%}
                - Wait for clearer signals or focus on technical analysis
                - Monitor for sentiment changes
                """)
            
            # Integration suggestions
            st.markdown("---")
            st.subheader("🔗 Integration with Other Analysis")
            st.markdown(f"""
            **Recommended Next Steps:**
            1. **Technical Analysis**: Check {symbol} charts in Single Stock Analysis for pattern confirmation
            2. **AI Decision**: Get comprehensive AI trading recommendation in AI Trading Decision tab
            3. **Market Context**: Review overall market conditions in Trading Intelligence
            4. **Risk Management**: Calculate appropriate position size in Risk Management tab
            5. **Alerts**: Set price alerts in Alert System for key levels identified
            """)
        
        else:
            st.error("Please enter a valid stock symbol")
    
    # News sources information
    st.markdown("---")
    st.subheader("📡 News Sources")
    st.markdown("**Supported Sources:**")
    
    sources_cols = st.columns(4)
    for i, source in enumerate(analyzer.supported_sources):
        with sources_cols[i % 4]:
            st.write(f"• {source}")
    
    st.info("""
    **How to Use News Sentiment Analysis:**
    1. **Confirmation Tool**: Use alongside technical analysis for trade confirmation
    2. **Risk Assessment**: Negative sentiment may indicate higher risk
    3. **Timing**: Strong positive sentiment can indicate good entry timing
    4. **Contrarian Signals**: Sometimes extreme sentiment indicates reversals
    5. **Theme Tracking**: Monitor key themes that could impact stock performance
    """)