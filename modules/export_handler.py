import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging
import streamlit as st
import xlsxwriter
from io import BytesIO

logger = logging.getLogger(__name__)

class ExportHandler:
    """Handles exporting pattern analysis results to various formats"""
    
    def __init__(self):
        self.formats = ['Excel', 'CSV']
    
    def create_patterns_dataframe(self, patterns: List[Dict[str, Any]], 
                                 data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create a comprehensive DataFrame with pattern analysis results"""
        
        if not patterns:
            return pd.DataFrame()
        
        export_data = []
        
        for pattern in patterns:
            try:
                # Get pattern candle data
                start_idx = pattern['start_idx']
                end_idx = pattern['end_idx']
                pattern_data = data.iloc[start_idx:end_idx+1]
                
                # Calculate pattern metrics
                pattern_info = {
                    'Symbol': symbol,
                    'Pattern_Type': pattern['type'],
                    'Date': pattern['date'],
                    'Confidence': f"{pattern['confidence']:.1%}",
                    'Signal': pattern.get('signal', 'Unknown'),
                    'Start_Date': pattern_data.index[0].strftime('%Y-%m-%d %H:%M'),
                    'End_Date': pattern_data.index[-1].strftime('%Y-%m-%d %H:%M'),
                    'Duration_Candles': len(pattern_data),
                    'Price_Range_Start': f"${pattern_data.iloc[0]['Open']:.2f}",
                    'Price_Range_End': f"${pattern_data.iloc[-1]['Close']:.2f}",
                    'High_During_Pattern': f"${pattern_data['High'].max():.2f}",
                    'Low_During_Pattern': f"${pattern_data['Low'].min():.2f}",
                    'Volume_Average': f"{pattern_data['Volume'].mean():,.0f}",
                    'Price_Change': f"{((pattern_data.iloc[-1]['Close'] - pattern_data.iloc[0]['Open']) / pattern_data.iloc[0]['Open'] * 100):.2f}%"
                }
                
                # Add individual candle details
                for i, (idx, row) in enumerate(pattern_data.iterrows()):
                    candle_info = pattern_info.copy()
                    candle_info.update({
                        'Candle_Position': i + 1,
                        'Candle_Date': idx.strftime('%Y-%m-%d %H:%M'),
                        'Open': f"${row['Open']:.2f}",
                        'High': f"${row['High']:.2f}",
                        'Low': f"${row['Low']:.2f}",
                        'Close': f"${row['Close']:.2f}",
                        'Volume': f"{row['Volume']:,.0f}",
                        'Body_Size': f"${abs(row['Close'] - row['Open']):.2f}",
                        'Upper_Shadow': f"${row['High'] - max(row['Open'], row['Close']):.2f}",
                        'Lower_Shadow': f"${min(row['Open'], row['Close']) - row['Low']:.2f}",
                        'Candle_Type': 'Bullish' if row['Close'] > row['Open'] else 'Bearish'
                    })
                    export_data.append(candle_info)
                    
            except Exception as e:
                logger.error(f"Error processing pattern {pattern.get('type', 'Unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(export_data)
    
    def export_to_excel(self, patterns: List[Dict[str, Any]], 
                       data: pd.DataFrame, symbol: str, timeframe: str) -> BytesIO:
        """Export pattern analysis to Excel with multiple sheets and formatting"""
        
        # Create BytesIO buffer
        buffer = BytesIO()
        
        try:
            # Create patterns DataFrame
            patterns_df = self.create_patterns_dataframe(patterns, data, symbol)
            
            # Create summary DataFrame
            summary_data = self._create_summary_data(patterns, symbol, timeframe)
            summary_df = pd.DataFrame([summary_data])
            
            # Create pattern frequency DataFrame
            frequency_df = self._create_frequency_data(patterns)
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Pattern frequency sheet
                frequency_df.to_excel(writer, sheet_name='Pattern_Frequency', index=False)
                
                # Detailed patterns sheet
                if not patterns_df.empty:
                    patterns_df.to_excel(writer, sheet_name='Detailed_Patterns', index=False)
                
                # Market data sheet (sample)
                data_sample = data.tail(100).copy()  # Last 100 candles
                data_sample.index = data_sample.index.strftime('%Y-%m-%d %H:%M')
                data_sample.to_excel(writer, sheet_name='Market_Data_Sample')
                
                # Get workbook and add formatting
                workbook = writer.book
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4CAF50',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Apply header formatting
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                
                # Format detailed patterns sheet if it exists
                if not patterns_df.empty:
                    patterns_sheet = writer.sheets['Detailed_Patterns']
                    for col_num, value in enumerate(patterns_df.columns.values):
                        patterns_sheet.write(0, col_num, value, header_format)
            
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Error creating Excel export: {str(e)}")
            return None
    
    def export_to_csv(self, patterns: List[Dict[str, Any]], 
                     data: pd.DataFrame, symbol: str) -> str:
        """Export pattern analysis to CSV format"""
        
        try:
            patterns_df = self.create_patterns_dataframe(patterns, data, symbol)
            
            if patterns_df.empty:
                return "No patterns found to export."
            
            # Convert to CSV string
            csv_string = patterns_df.to_csv(index=False)
            return csv_string
            
        except Exception as e:
            logger.error(f"Error creating CSV export: {str(e)}")
            return None
    
    def _create_summary_data(self, patterns: List[Dict[str, Any]], 
                           symbol: str, timeframe: str) -> Dict[str, Any]:
        """Create summary data for export"""
        
        if not patterns:
            return {
                'Symbol': symbol,
                'Timeframe': timeframe,
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Total_Patterns': 0,
                'Bullish_Patterns': 0,
                'Bearish_Patterns': 0,
                'Neutral_Patterns': 0,
                'Average_Confidence': '0%',
                'Most_Common_Pattern': 'None'
            }
        
        # Count pattern types
        pattern_counts = {}
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_confidence = 0
        
        for pattern in patterns:
            pattern_type = pattern['type']
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            signal = pattern.get('signal', '')
            if 'Bullish' in signal:
                bullish_count += 1
            elif 'Bearish' in signal:
                bearish_count += 1
            else:
                neutral_count += 1
                
            total_confidence += pattern['confidence']
        
        most_common = max(pattern_counts, key=pattern_counts.get) if pattern_counts else 'None'
        avg_confidence = total_confidence / len(patterns) if patterns else 0
        
        return {
            'Symbol': symbol,
            'Timeframe': timeframe,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'Total_Patterns': len(patterns),
            'Bullish_Patterns': bullish_count,
            'Bearish_Patterns': bearish_count,
            'Neutral_Patterns': neutral_count,
            'Average_Confidence': f"{avg_confidence:.1%}",
            'Most_Common_Pattern': most_common
        }
    
    def _create_frequency_data(self, patterns: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create pattern frequency data"""
        
        if not patterns:
            return pd.DataFrame({'Pattern_Type': [], 'Count': [], 'Percentage': []})
        
        # Count patterns
        pattern_counts = {}
        for pattern in patterns:
            pattern_type = pattern['type']
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        total_patterns = len(patterns)
        frequency_data = []
        
        for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_patterns) * 100
            frequency_data.append({
                'Pattern_Type': pattern_type,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        return pd.DataFrame(frequency_data)
    
    def create_download_button(self, patterns: List[Dict[str, Any]], 
                             data: pd.DataFrame, symbol: str, 
                             timeframe: str, export_format: str) -> bool:
        """Create appropriate download button based on format"""
        
        if not patterns:
            st.warning("No patterns found to export.")
            return False
        
        try:
            if export_format == 'Excel':
                excel_buffer = self.export_to_excel(patterns, data, symbol, timeframe)
                if excel_buffer:
                    filename = f"{symbol}_{timeframe}_patterns_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    return True
                    
            elif export_format == 'CSV':
                csv_data = self.export_to_csv(patterns, data, symbol)
                if csv_data:
                    filename = f"{symbol}_{timeframe}_patterns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                    return True
            
            # Add enhanced platform download option
            st.markdown("---")
            st.markdown("**📦 Enhanced Platform Download**")
            
            if st.button("Download Enhanced Trading Platform", help="Download the complete enhanced trading platform with latest fixes"):
                try:
                    with open('trading_platform_enhanced.zip', 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Enhanced Platform ZIP",
                            data=f.read(),
                            file_name=f'trading_platform_enhanced_{timeframe}_{datetime.now().strftime("%Y%m%d")}.zip',
                            mime='application/zip',
                            key='enhanced_platform_download',
                            use_container_width=True
                        )
                    st.success("Enhanced platform download ready!")
                    return True
                except FileNotFoundError:
                    # Fallback to original platform
                    try:
                        with open('trading_platform_complete.zip', 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Original Platform ZIP",
                                data=f.read(),
                                file_name=f'trading_platform_complete_{timeframe}.zip',
                                mime='application/zip',
                                key='platform_download',
                                use_container_width=True
                            )
                        st.success("Original platform download ready!")
                        return True
                    except FileNotFoundError:
                        st.error("Platform ZIP file not found. Please contact support.")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating download button: {str(e)}")
            st.error(f"Error preparing download: {str(e)}")
            return False