import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class MarketScanner:
    """Scan multiple stocks for candlestick patterns and trading opportunities"""
    
    def __init__(self):
        self.popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA',
            'JPM', 'JNJ', 'UNH', 'PG', 'HD', 'V', 'MA', 'DIS', 'PYPL', 'BAC',
            'ADBE', 'CRM', 'INTC', 'CSCO', 'PFE', 'WMT', 'KO', 'PEP', 'ABT',
            'TMO', 'COST', 'AVGO', 'TXN', 'QCOM', 'HON', 'NEE', 'UPS', 'LOW'
        ]
        
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD',
            'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD'
        ]
        
        self.forex_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
            'USDCHF=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
        ]
    
    def scan_watchlist(self, symbols: List[str], pattern_recognition, timeframe: str = '1d', 
                      max_workers: int = 5) -> List[Dict[str, Any]]:
        """Scan a list of symbols for patterns"""
        results = []
        
        def scan_single_symbol(symbol):
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)  # Last 60 days
                
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=self._convert_timeframe(timeframe)
                )
                
                if data.empty or len(data) < 20:
                    return None
                
                # Add technical indicators
                data = self._add_basic_indicators(data)
                
                # Analyze patterns
                patterns = pattern_recognition.analyze_patterns(data, pattern_recognition.patterns.keys())
                
                if patterns:
                    # Get latest price info
                    latest = data.iloc[-1]
                    change = ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) * 100
                    
                    return {
                        'symbol': symbol,
                        'current_price': latest['Close'],
                        'change_percent': change,
                        'volume': latest['Volume'],
                        'patterns_found': len(patterns),
                        'latest_pattern': patterns[-1] if patterns else None,
                        'patterns': patterns[-3:],  # Last 3 patterns
                        'last_updated': datetime.now().strftime('%H:%M:%S')
                    }
                
                return None
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {str(e)}")
                return None
        
        # Use ThreadPoolExecutor for concurrent scanning
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(scan_single_symbol, symbol): symbol 
                              for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort by patterns found and recent activity
        results.sort(key=lambda x: (x['patterns_found'], abs(x['change_percent'])), reverse=True)
        
        return results
    
    def get_market_movers(self, min_volume: int = 1000000, min_change: float = 2.0) -> List[Dict[str, Any]]:
        """Get stocks with significant price movements and volume"""
        movers = []
        
        for symbol in self.popular_stocks[:20]:  # Check top 20 for speed
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2d', interval='1d')
                
                if len(data) >= 2:
                    latest = data.iloc[-1]
                    previous = data.iloc[-2]
                    
                    change_percent = ((latest['Close'] - previous['Close']) / previous['Close']) * 100
                    
                    if abs(change_percent) >= min_change and latest['Volume'] >= min_volume:
                        movers.append({
                            'symbol': symbol,
                            'price': latest['Close'],
                            'change_percent': change_percent,
                            'volume': latest['Volume'],
                            'direction': 'UP' if change_percent > 0 else 'DOWN'
                        })
                        
            except Exception as e:
                logger.error(f"Error getting mover data for {symbol}: {str(e)}")
                continue
        
        return sorted(movers, key=lambda x: abs(x['change_percent']), reverse=True)
    
    def scan_for_breakouts(self, symbols: List[str], lookback_period: int = 20) -> List[Dict[str, Any]]:
        """Scan for potential breakout patterns"""
        breakouts = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo', interval='1d')
                
                if len(data) < lookback_period + 5:
                    continue
                
                # Calculate recent highs and lows
                recent_high = data['High'].rolling(window=lookback_period).max()
                recent_low = data['Low'].rolling(window=lookback_period).min()
                
                latest = data.iloc[-1]
                previous_high = recent_high.iloc[-2]
                previous_low = recent_low.iloc[-2]
                
                # Check for breakouts
                if latest['Close'] > previous_high:
                    breakout_type = 'Upward Breakout'
                    strength = (latest['Close'] - previous_high) / previous_high * 100
                elif latest['Close'] < previous_low:
                    breakout_type = 'Downward Breakout'
                    strength = (previous_low - latest['Close']) / previous_low * 100
                else:
                    continue
                
                # Volume confirmation
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                volume_ratio = latest['Volume'] / avg_volume
                
                if volume_ratio > 1.5:  # Volume above average
                    breakouts.append({
                        'symbol': symbol,
                        'type': breakout_type,
                        'current_price': latest['Close'],
                        'breakout_level': previous_high if 'Upward' in breakout_type else previous_low,
                        'strength_percent': strength,
                        'volume_ratio': volume_ratio,
                        'confidence': min(0.95, strength * volume_ratio / 10)
                    })
                    
            except Exception as e:
                logger.error(f"Error scanning breakouts for {symbol}: {str(e)}")
                continue
        
        return sorted(breakouts, key=lambda x: x['confidence'], reverse=True)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to yfinance format"""
        mapping = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '1hour': '1h',
            'daily': '1d'
        }
        return mapping.get(timeframe, '1d')
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to data"""
        data = data.copy()
        
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        # Body size and shadows
        data['Body_Size'] = abs(data['Close'] - data['Open'])
        data['Upper_Shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        data['Lower_Shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        
        return data
    
    def create_watchlist_dashboard(self, watchlist_results: List[Dict[str, Any]]) -> None:
        """Create a dashboard display for watchlist results"""
        
        if not watchlist_results:
            st.info("No patterns found in the scanned symbols.")
            return
        
        st.subheader("📊 Market Scanner Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbols Scanned", len(watchlist_results))
        
        with col2:
            total_patterns = sum(r['patterns_found'] for r in watchlist_results)
            st.metric("Total Patterns", total_patterns)
        
        with col3:
            bullish_count = sum(1 for r in watchlist_results 
                              if r['latest_pattern'] and 'Bullish' in r['latest_pattern'].get('signal', ''))
            st.metric("Bullish Signals", bullish_count)
        
        with col4:
            bearish_count = sum(1 for r in watchlist_results 
                              if r['latest_pattern'] and 'Bearish' in r['latest_pattern'].get('signal', ''))
            st.metric("Bearish Signals", bearish_count)
        
        # Results table
        df_results = pd.DataFrame([
            {
                'Symbol': r['symbol'],
                'Price': f"${r['current_price']:.2f}",
                'Change %': f"{r['change_percent']:+.2f}%",
                'Volume': f"{r['volume']:,.0f}",
                'Patterns': r['patterns_found'],
                'Latest Pattern': r['latest_pattern']['type'] if r['latest_pattern'] else 'None',
                'Signal': r['latest_pattern']['signal'] if r['latest_pattern'] else 'None',
                'Confidence': f"{r['latest_pattern']['confidence']:.1%}" if r['latest_pattern'] else 'N/A'
            }
            for r in watchlist_results
        ])
        
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True
        )
        
        # Detailed pattern information
        st.subheader("🔍 Detailed Pattern Information")
        
        for result in watchlist_results[:10]:  # Show top 10
            if result['latest_pattern']:
                with st.expander(f"{result['symbol']} - {result['latest_pattern']['type']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Current Price:** ${result['current_price']:.2f}")
                        st.write(f"**Change:** {result['change_percent']:+.2f}%")
                        st.write(f"**Volume:** {result['volume']:,}")
                    
                    with col2:
                        pattern = result['latest_pattern']
                        st.write(f"**Pattern:** {pattern['type']}")
                        st.write(f"**Signal:** {pattern['signal']}")
                        st.write(f"**Confidence:** {pattern['confidence']:.1%}")
                        st.write(f"**Date:** {pattern['date']}")
    
    def get_sector_analysis(self) -> Dict[str, List[str]]:
        """Get stocks categorized by sector for focused scanning"""
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX', 'NVDA', 'ADBE', 'CRM', 'INTC', 'CSCO'],
            'Finance': ['JPM', 'BAC', 'V', 'MA', 'PYPL', 'GS', 'WFC', 'C', 'AXP', 'BLK'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'CVS', 'ABBV', 'MDT', 'BMY'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'DIS', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI'],
            'Crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD']
        }
        return sectors