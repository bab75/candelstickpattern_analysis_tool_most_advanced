# 📈 Candlestick Pattern Analysis Tool

A comprehensive, modular candlestick pattern analysis tool built with Python and Streamlit. This application provides real-time market data analysis, interactive visualizations, and educational content about candlestick patterns for traders and investors.

## 🚀 Features

### Core Functionality
- **Real-time Data Analysis**: Live market data updates during trading hours
- **Historical Analysis**: Analyze patterns across custom date ranges
- **Multiple Timeframes**: Support for 1min, 5min, 15min, 1hour, and daily charts
- **Pattern Recognition**: Automated detection of 13+ major candlestick patterns
- **Interactive Charts**: Plotly-powered candlestick visualizations with pattern annotations
- **Educational Content**: Detailed explanations of pattern formation and trading significance

### Supported Candlestick Patterns
- **Reversal Patterns**: Hammer, Inverted Hammer, Shooting Star, Hanging Man, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Pattern, Dark Cloud Cover
- **Continuation Patterns**: Three White Soldiers, Three Black Crows
- **Indecision Patterns**: Doji variations

### Technical Features
- Market session indicators and status
- Volume analysis and confirmation
- Moving averages (SMA 20, SMA 50)
- Support and resistance level identification
- Pattern confidence scoring
- Gap detection
- Volatility calculations

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Source**: Yahoo Finance (yfinance)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Local Development

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/candlestick-analysis-tool.git
cd candlestick-analysis-tool
