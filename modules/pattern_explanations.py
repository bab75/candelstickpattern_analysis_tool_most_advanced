from typing import Dict, Any

class PatternExplanations:
    """Provides detailed explanations for candlestick patterns"""
    
    def __init__(self):
        self.explanations = {
            'Hammer': {
                'description': 'A bullish reversal pattern that forms at the bottom of a downtrend. It has a small body at the upper end of the trading range with a long lower shadow.',
                'formation': 'The hammer forms when sellers push the price down significantly during the session, but buyers step in and drive the price back up, closing near or above the opening price. The long lower shadow shows rejection of lower prices.',
                'significance': 'Indicates potential bullish reversal as buyers have shown strength by rejecting lower prices. Most effective when appearing after a prolonged downtrend.',
                'strategy': 'Wait for confirmation with a bullish candle the next session. Consider long positions with stop loss below the hammer\'s low. Target previous resistance levels.'
            },
            'Inverted Hammer': {
                'description': 'A potential bullish reversal pattern with a small body at the lower end and a long upper shadow, appearing at the bottom of downtrends.',
                'formation': 'Buyers push prices higher during the session, but sellers regain control and push prices back down near the opening. Despite the bearish close, it shows buying interest.',
                'significance': 'Suggests buyers are starting to challenge the downtrend. Requires confirmation from subsequent price action to validate the reversal.',
                'strategy': 'Look for bullish confirmation in the next session. Enter long positions on confirmation with stops below the pattern low.'
            },
            'Shooting Star': {
                'description': 'A bearish reversal pattern that appears at the top of uptrends, characterized by a small body at the lower end with a long upper shadow.',
                'formation': 'Buyers initially push prices higher, but sellers take control and drive prices back down, closing near the session low. Shows rejection of higher prices.',
                'significance': 'Indicates potential trend reversal as higher prices were rejected. Most reliable when appearing after a significant uptrend.',
                'strategy': 'Wait for bearish confirmation. Consider short positions or profit-taking on long positions. Set stops above the shooting star\'s high.'
            },
            'Hanging Man': {
                'description': 'A bearish reversal pattern similar to a hammer but appears at the top of uptrends, featuring a small body and long lower shadow.',
                'formation': 'Despite the bullish appearance, the long lower shadow at the top of an uptrend suggests selling pressure. Buyers initially supported the price, but the context makes it bearish.',
                'significance': 'Warns of potential trend exhaustion and possible reversal. The hanging position suggests the uptrend may be in jeopardy.',
                'strategy': 'Monitor for bearish confirmation in subsequent sessions. Consider reducing long positions or preparing for potential reversal.'
            },
            'Doji': {
                'description': 'An indecision pattern where opening and closing prices are virtually equal, creating a cross-like appearance with shadows extending from both sides.',
                'formation': 'Neither bulls nor bears can gain control during the session, resulting in a close near the open. The battle between buyers and sellers ends in a stalemate.',
                'significance': 'Represents market indecision and potential trend change. More significant when appearing after strong trending moves.',
                'strategy': 'Wait for direction confirmation in subsequent candles. Avoid taking new positions until market direction becomes clear.'
            },
            'Bullish Engulfing': {
                'description': 'A two-candle bullish reversal pattern where a large bullish candle completely engulfs the previous bearish candle\'s body.',
                'formation': 'First candle is bearish, followed by a bullish candle that opens below the previous close but closes above the previous open, completely engulfing the first candle\'s body.',
                'significance': 'Strong bullish reversal signal indicating buyers have overwhelmed sellers. Shows significant shift in market sentiment.',
                'strategy': 'Enter long positions on the engulfing candle or on a pullback. Set stops below the pattern low. Target previous resistance levels.'
            },
            'Bearish Engulfing': {
                'description': 'A two-candle bearish reversal pattern where a large bearish candle completely engulfs the previous bullish candle\'s body.',
                'formation': 'First candle is bullish, followed by a bearish candle that opens above the previous close but closes below the previous open, engulfing the entire first body.',
                'significance': 'Strong bearish reversal signal showing sellers have taken control. Indicates significant downward momentum.',
                'strategy': 'Consider short positions or exit long positions. Set stops above the pattern high. Target previous support levels.'
            },
            'Morning Star': {
                'description': 'A three-candle bullish reversal pattern consisting of a bearish candle, a small-bodied candle (star), and a bullish candle.',
                'formation': 'First candle is bearish in a downtrend, second is a small-bodied candle showing indecision, third is bullish and closes well into the first candle\'s body.',
                'significance': 'Powerful bullish reversal pattern indicating the end of a downtrend. The star shows indecision, while the third candle confirms bullish momentum.',
                'strategy': 'Enter long positions on the third candle or on confirmation. Set stops below the star\'s low. Target previous resistance levels.'
            },
            'Evening Star': {
                'description': 'A three-candle bearish reversal pattern with a bullish candle, a small-bodied star, and a bearish candle.',
                'formation': 'First candle is bullish in an uptrend, second is a small star showing indecision, third is bearish and closes well into the first candle\'s body.',
                'significance': 'Strong bearish reversal pattern marking potential end of uptrend. Shows shift from bullish to bearish sentiment.',
                'strategy': 'Consider short positions or profit-taking. Set stops above the star\'s high. Target key support levels.'
            },
            'Three White Soldiers': {
                'description': 'A bullish continuation pattern consisting of three consecutive bullish candles with progressively higher closes.',
                'formation': 'Three bullish candles in succession, each opening within the previous candle\'s body and closing higher. Shows sustained buying pressure.',
                'significance': 'Strong bullish continuation signal indicating persistent buying interest. Suggests the uptrend will likely continue.',
                'strategy': 'Consider adding to long positions or entering new long positions. Trail stops below recent lows. Target next resistance level.'
            },
            'Three Black Crows': {
                'description': 'A bearish continuation pattern with three consecutive bearish candles showing progressively lower closes.',
                'formation': 'Three bearish candles in sequence, each opening within the previous candle\'s body and closing lower. Demonstrates sustained selling pressure.',
                'significance': 'Strong bearish continuation signal indicating persistent selling. Suggests the downtrend will likely continue.',
                'strategy': 'Consider short positions or exit long positions. Set stops above recent highs. Target next support level.'
            },
            'Piercing Pattern': {
                'description': 'A two-candle bullish reversal pattern where a bullish candle pierces more than halfway into the previous bearish candle.',
                'formation': 'First candle is bearish, second opens below the first\'s low but closes above the midpoint of the first candle\'s body.',
                'significance': 'Bullish reversal signal showing buyers stepping in aggressively. The deeper the penetration, the stronger the signal.',
                'strategy': 'Enter long positions on confirmation. Set stops below the pattern low. Target previous resistance areas.'
            },
            'Dark Cloud Cover': {
                'description': 'A two-candle bearish reversal pattern where a bearish candle opens above and closes below the midpoint of the previous bullish candle.',
                'formation': 'First candle is bullish, second opens above the first\'s high but closes below the midpoint of the first candle\'s body.',
                'significance': 'Bearish reversal signal indicating selling pressure. The deeper the penetration below the midpoint, the more bearish the signal.',
                'strategy': 'Consider short positions or reduce long exposure. Set stops above the pattern high. Target support levels.'
            }
        }
    
    def get_explanation(self, pattern_type: str) -> Dict[str, str]:
        """
        Get detailed explanation for a specific pattern type
        
        Args:
            pattern_type: Name of the candlestick pattern
            
        Returns:
            Dictionary with pattern explanation details
        """
        return self.explanations.get(pattern_type, {
            'description': f'Information for {pattern_type} pattern is not available.',
            'formation': 'Pattern formation details not available.',
            'significance': 'Trading significance not available.',
            'strategy': 'Trading strategy not available.'
        })
    
    def get_all_patterns(self) -> Dict[str, Dict[str, str]]:
        """Get explanations for all available patterns"""
        return self.explanations
    
    def get_pattern_categories(self) -> Dict[str, list]:
        """Categorize patterns by their signal type"""
        categories = {
            'Bullish Reversal': [],
            'Bearish Reversal': [],
            'Continuation': [],
            'Indecision': []
        }
        
        reversal_bullish = [
            'Hammer', 'Inverted Hammer', 'Bullish Engulfing', 
            'Morning Star', 'Piercing Pattern'
        ]
        
        reversal_bearish = [
            'Shooting Star', 'Hanging Man', 'Bearish Engulfing',
            'Evening Star', 'Dark Cloud Cover'
        ]
        
        continuation = ['Three White Soldiers', 'Three Black Crows']
        indecision = ['Doji']
        
        categories['Bullish Reversal'] = reversal_bullish
        categories['Bearish Reversal'] = reversal_bearish
        categories['Continuation'] = continuation
        categories['Indecision'] = indecision
        
        return categories
    
    def get_trading_tips(self) -> Dict[str, str]:
        """Get general trading tips for candlestick patterns"""
        return {
            'Confirmation': 'Always wait for confirmation before acting on pattern signals. A single pattern should not be the sole basis for trading decisions.',
            'Volume': 'Look for volume confirmation. Patterns with higher volume tend to be more reliable.',
            'Context': 'Consider the overall trend and market context. Reversal patterns are more significant at trend extremes.',
            'Risk Management': 'Always use stop losses and proper position sizing. Patterns can fail, so manage your risk accordingly.',
            'Multiple Timeframes': 'Analyze patterns across multiple timeframes for better confirmation and timing.',
            'Market Conditions': 'Pattern reliability can vary based on market volatility and liquidity conditions.',
            'Combine Indicators': 'Use patterns in conjunction with other technical indicators for better signal quality.'
        }
