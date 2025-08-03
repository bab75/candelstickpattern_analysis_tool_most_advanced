import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generates interactive candlestick charts with pattern annotations"""
    
    def __init__(self):
        self.color_scheme = {
            'bullish': '#26a69a',
            'bearish': '#ef5350',
            'volume': '#1f77b4',
            'sma_20': '#ff7f0e',
            'sma_50': '#2ca02c',
            'pattern_bull': '#4caf50',
            'pattern_bear': '#f44336',
            'pattern_neutral': '#ff9800'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, patterns: List[Dict[str, Any]], 
                               symbol: str, timeframe: str) -> go.Figure:
        """
        Create interactive candlestick chart with pattern annotations
        
        Args:
            data: OHLCV DataFrame
            patterns: List of detected patterns
            symbol: Stock symbol
            timeframe: Chart timeframe
            
        Returns:
            Plotly figure object
        """
        try:
            # Create subplots with secondary y-axis for volume
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[f'{symbol} - {timeframe} Candlestick Chart', 'Volume'],
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol,
                    increasing_line_color=self.color_scheme['bullish'],
                    decreasing_line_color=self.color_scheme['bearish'],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color=self.color_scheme['sma_20'], width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color=self.color_scheme['sma_50'], width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Add volume bars
            colors = [self.color_scheme['bullish'] if close >= open_price 
                     else self.color_scheme['bearish'] 
                     for close, open_price in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add pattern annotations
            self._add_pattern_annotations(fig, data, patterns)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Candlestick Analysis - {timeframe}',
                template='plotly_white',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        text="💡 Hover over patterns for details | Zoom and pan for better analysis",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.1,
                        xanchor='center', yanchor='top',
                        font=dict(size=12, color="gray")
                    )
                ]
            )
            
            # Update x-axis
            fig.update_xaxes(
                title_text="Time",
                row=2, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(
                title_text="Price ($)",
                row=1, col=1
            )
            fig.update_yaxes(
                title_text="Volume",
                row=2, col=1
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
    
    def _add_pattern_annotations(self, fig: go.Figure, data: pd.DataFrame, 
                               patterns: List[Dict[str, Any]]) -> None:
        """Add pattern annotations to the chart"""
        
        pattern_colors = {
            'Bullish Reversal': self.color_scheme['pattern_bull'],
            'Bearish Reversal': self.color_scheme['pattern_bear'],
            'Strong Bullish Continuation': self.color_scheme['pattern_bull'],
            'Strong Bearish Continuation': self.color_scheme['pattern_bear'],
            'Potential Bullish Reversal': self.color_scheme['pattern_bull'],
            'Market Indecision': self.color_scheme['pattern_neutral']
        }
        
        for pattern in patterns:
            try:
                start_idx = pattern['start_idx']
                end_idx = pattern['end_idx']
                
                # Get pattern data
                pattern_data = data.iloc[start_idx:end_idx+1]
                
                # Determine annotation position
                if len(pattern_data) > 0:
                    x_pos = pattern_data.index[-1]  # Use last candle position
                    y_pos = pattern_data['High'].max() * 1.02  # Position above highest point
                    
                    signal_type = pattern.get('signal', 'Unknown')
                    color = pattern_colors.get(signal_type, self.color_scheme['pattern_neutral'])
                    
                    # Add pattern marker
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos],
                            y=[y_pos],
                            mode='markers+text',
                            marker=dict(
                                size=12,
                                color=color,
                                symbol='star',
                                line=dict(width=2, color='white')
                            ),
                            text=[pattern['type']],
                            textposition='top center',
                            textfont=dict(size=10, color=color),
                            name=f"{pattern['type']} ({pattern['confidence']:.1%})",
                            hovertemplate=(
                                f"<b>{pattern['type']}</b><br>"
                                f"Date: {pattern['date']}<br>"
                                f"Signal: {signal_type}<br>"
                                f"Confidence: {pattern['confidence']:.1%}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Highlight pattern candles
                    for idx in range(start_idx, end_idx + 1):
                        if idx < len(data):
                            candle_data = data.iloc[idx]
                            fig.add_shape(
                                type="rect",
                                x0=candle_data.name,
                                x1=candle_data.name,
                                y0=candle_data['Low'] * 0.999,
                                y1=candle_data['High'] * 1.001,
                                line=dict(color=color, width=2),
                                fillcolor=color,
                                opacity=0.1,
                                row=1, col=1
                            )
                            
            except Exception as e:
                logger.warning(f"Error adding annotation for pattern {pattern.get('type', 'Unknown')}: {str(e)}")
                continue
    
    def create_pattern_detail_chart(self, data: pd.DataFrame, pattern: Dict[str, Any]) -> go.Figure:
        """Create detailed chart focused on a specific pattern"""
        
        start_idx = max(0, pattern['start_idx'] - 5)
        end_idx = min(len(data) - 1, pattern['end_idx'] + 5)
        
        pattern_data = data.iloc[start_idx:end_idx + 1]
        
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(
            go.Candlestick(
                x=pattern_data.index,
                open=pattern_data['Open'],
                high=pattern_data['High'],
                low=pattern_data['Low'],
                close=pattern_data['Close'],
                name='Candlesticks',
                increasing_line_color=self.color_scheme['bullish'],
                decreasing_line_color=self.color_scheme['bearish']
            )
        )
        
        # Highlight pattern candles
        pattern_candles = data.iloc[pattern['start_idx']:pattern['end_idx'] + 1]
        for idx, (timestamp, candle) in enumerate(pattern_candles.iterrows()):
            fig.add_shape(
                type="rect",
                x0=timestamp,
                x1=timestamp,
                y0=candle['Low'] * 0.999,
                y1=candle['High'] * 1.001,
                line=dict(color="gold", width=3),
                fillcolor="gold",
                opacity=0.2
            )
        
        fig.update_layout(
            title=f"{pattern['type']} Pattern Detail - {pattern['date']}",
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=400
        )
        
        return fig
