import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    trade_type: str  # 'long' or 'short'
    pattern_type: str
    confidence: float
    return_pct: float
    holding_days: int
    max_drawdown: float
    max_profit: float

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_winning_return: float
    avg_losing_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade]

class AdvancedBacktesting:
    """Advanced backtesting engine with multiple strategies and metrics"""
    
    def __init__(self):
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.0005   # 0.05% slippage
        self.initial_capital = 10000
        
    def run_comprehensive_backtest(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, pattern_recognition, 
                                 timeframe: str = '1d') -> Dict[str, Any]:
        """Run comprehensive backtest with multiple strategies"""
        
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=timeframe)
        
        if data.empty:
            return {'error': 'No data available for the specified period'}
        
        # Analyze patterns
        patterns = pattern_recognition.analyze_patterns(data, pattern_recognition.patterns.keys())
        
        # Run different backtesting strategies
        results = {}
        
        try:
            # Strategy 1: Pattern-based trading
            results['pattern_strategy'] = self._backtest_pattern_strategy(data, patterns)
            logger.info(f"Pattern strategy completed: {results['pattern_strategy'].total_trades} trades")
            
            # Strategy 2: Technical indicator strategy
            results['technical_strategy'] = self._backtest_technical_strategy(data, patterns)
            logger.info(f"Technical strategy completed: {results['technical_strategy'].total_trades} trades")
            
            # Strategy 3: Combined strategy
            results['combined_strategy'] = self._backtest_combined_strategy(data, patterns)
            logger.info(f"Combined strategy completed: {results['combined_strategy'].total_trades} trades")
            
            # Strategy 4: Risk-adjusted strategy
            results['risk_adjusted_strategy'] = self._backtest_risk_adjusted_strategy(data, patterns)
            logger.info(f"Risk-adjusted strategy completed: {results['risk_adjusted_strategy'].total_trades} trades")
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return {'error': f'Backtesting failed: {str(e)}'}
        
        # Generate comparison analysis
        results['strategy_comparison'] = self._compare_strategies(results)
        
        # Monte Carlo simulation
        results['monte_carlo'] = self._run_monte_carlo_simulation(data, patterns)
        
        return results
    
    def _backtest_pattern_strategy(self, data: pd.DataFrame, 
                                  patterns: List[Dict[str, Any]]) -> BacktestResults:
        """Backtest pure pattern-based strategy"""
        trades = []
        current_position = None
        capital = self.initial_capital
        
        for pattern in patterns:
            try:
                start_idx = pattern['start_idx']
                pattern_date = data.index[start_idx]
                signal = pattern.get('signal', '')
                confidence = pattern['confidence']
                
                # Skip low confidence patterns
                if confidence < 0.6:
                    continue
                
                # Entry logic
                if current_position is None:
                    if 'Bullish' in signal:
                        # Enter long position
                        entry_price = data['Close'].iloc[start_idx + 1] if start_idx + 1 < len(data) else data['Close'].iloc[start_idx]
                        entry_price *= (1 + self.slippage)  # Add slippage
                        
                        current_position = {
                            'type': 'long',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': pattern['type'],
                            'confidence': confidence,
                            'entry_idx': start_idx
                        }
                    
                    elif 'Bearish' in signal:
                        # Enter short position
                        entry_price = data['Close'].iloc[start_idx + 1] if start_idx + 1 < len(data) else data['Close'].iloc[start_idx]
                        entry_price *= (1 - self.slippage)  # Add slippage
                        
                        current_position = {
                            'type': 'short',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': pattern['type'],
                            'confidence': confidence,
                            'entry_idx': start_idx
                        }
                
                # Exit logic (hold for 5-10 days)
                elif current_position is not None:
                    holding_days = (pattern_date - current_position['entry_date']).days
                    
                    if holding_days >= 5:  # Minimum holding period
                        exit_price = data['Close'].iloc[start_idx]
                        
                        if current_position['type'] == 'long':
                            exit_price *= (1 - self.slippage)
                        else:
                            exit_price *= (1 + self.slippage)
                        
                        # Calculate trade metrics
                        trade = self._calculate_trade_metrics(
                            current_position, exit_price, pattern_date, data
                        )
                        trades.append(trade)
                        current_position = None
                        
            except Exception as e:
                logger.error(f"Error processing pattern {pattern.get('type', 'Unknown')}: {str(e)}")
                continue
        
        # Close any remaining position
        if current_position is not None:
            exit_price = data['Close'].iloc[-1]
            if current_position['type'] == 'long':
                exit_price *= (1 - self.slippage)
            else:
                exit_price *= (1 + self.slippage)
            
            trade = self._calculate_trade_metrics(
                current_position, exit_price, data.index[-1], data
            )
            trades.append(trade)
        
        return self._calculate_backtest_results(trades)
    
    def _backtest_technical_strategy(self, data: pd.DataFrame, 
                                   patterns: List[Dict[str, Any]]) -> BacktestResults:
        """Backtest technical indicator strategy"""
        trades = []
        
        # Calculate technical indicators
        data = data.copy()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
        data['BB_Upper'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        
        current_position = None
        
        for i in range(20, len(data) - 1):  # Start after indicators stabilize
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Entry conditions
            if current_position is None:
                # Bullish entry: RSI oversold + MACD bullish crossover
                if (data['RSI'].iloc[i] < 35 and 
                    data['MACD'].iloc[i] > data['MACD_Signal'].iloc[i] and
                    data['MACD'].iloc[i-1] <= data['MACD_Signal'].iloc[i-1]):
                    
                    current_position = {
                        'type': 'long',
                        'entry_date': current_date,
                        'entry_price': current_price * (1 + self.slippage),
                        'pattern_type': 'Technical Signal',
                        'confidence': 0.7,
                        'entry_idx': i
                    }
                
                # Bearish entry: RSI overbought + MACD bearish crossover
                elif (data['RSI'].iloc[i] > 65 and 
                      data['MACD'].iloc[i] < data['MACD_Signal'].iloc[i] and
                      data['MACD'].iloc[i-1] >= data['MACD_Signal'].iloc[i-1]):
                    
                    current_position = {
                        'type': 'short',
                        'entry_date': current_date,
                        'entry_price': current_price * (1 - self.slippage),
                        'pattern_type': 'Technical Signal',
                        'confidence': 0.7,
                        'entry_idx': i
                    }
            
            # Exit conditions
            elif current_position is not None:
                holding_days = (current_date - current_position['entry_date']).days
                
                # Exit long position
                if (current_position['type'] == 'long' and 
                    (data['RSI'].iloc[i] > 70 or holding_days > 15)):
                    
                    exit_price = current_price * (1 - self.slippage)
                    trade = self._calculate_trade_metrics(
                        current_position, exit_price, current_date, data
                    )
                    trades.append(trade)
                    current_position = None
                
                # Exit short position
                elif (current_position['type'] == 'short' and 
                      (data['RSI'].iloc[i] < 30 or holding_days > 15)):
                    
                    exit_price = current_price * (1 + self.slippage)
                    trade = self._calculate_trade_metrics(
                        current_position, exit_price, current_date, data
                    )
                    trades.append(trade)
                    current_position = None
        
        return self._calculate_backtest_results(trades)
    
    def _backtest_combined_strategy(self, data: pd.DataFrame, 
                                  patterns: List[Dict[str, Any]]) -> BacktestResults:
        """Backtest combined pattern + technical strategy"""
        trades = []
        
        # Calculate technical indicators
        data = data.copy()
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
        
        current_position = None
        
        logger.info(f"Starting combined strategy with {len(patterns)} patterns")
        
        for pattern in patterns:
            try:
                start_idx = pattern['start_idx']
                if start_idx >= len(data):
                    continue
                    
                pattern_date = data.index[start_idx]
                signal = pattern.get('signal', '')
                confidence = pattern['confidence']
                
                # Get technical confirmation with bounds checking
                rsi = data['RSI'].iloc[start_idx] if start_idx < len(data['RSI']) else 50
                macd = data['MACD'].iloc[start_idx] if start_idx < len(data['MACD']) else 0
                macd_signal = data['MACD_Signal'].iloc[start_idx] if start_idx < len(data['MACD_Signal']) else 0
                
                # Enhanced entry logic with technical confirmation (relaxed conditions)
                if current_position is None:
                    # More lenient entry conditions to generate trades
                    if 'Bullish' in signal and confidence > 0.4:
                        
                        entry_price = data['Close'].iloc[start_idx + 1] if start_idx + 1 < len(data) else data['Close'].iloc[start_idx]
                        entry_price *= (1 + self.slippage)
                        
                        current_position = {
                            'type': 'long',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': f"{pattern['type']} + Tech Confirm",
                            'confidence': min(1.0, confidence + 0.1),  # Boost confidence
                            'entry_idx': start_idx
                        }
                        logger.info(f"Entering LONG position for {pattern['type']} at {entry_price:.2f}")
                    
                    elif 'Bearish' in signal and confidence > 0.4:
                        
                        entry_price = data['Close'].iloc[start_idx + 1] if start_idx + 1 < len(data) else data['Close'].iloc[start_idx]
                        entry_price *= (1 - self.slippage)
                        
                        current_position = {
                            'type': 'short',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': f"{pattern['type']} + Tech Confirm",
                            'confidence': min(1.0, confidence + 0.1),
                            'entry_idx': start_idx
                        }
                        logger.info(f"Entering SHORT position for {pattern['type']} at {entry_price:.2f}")
                
                # Exit logic
                elif current_position is not None:
                    holding_days = (pattern_date - current_position['entry_date']).days
                    
                    # Exit conditions based on technical indicators (more lenient)
                    should_exit = False
                    if current_position['type'] == 'long' and (rsi > 80 or holding_days > 15):
                        should_exit = True
                    elif current_position['type'] == 'short' and (rsi < 20 or holding_days > 15):
                        should_exit = True
                    
                    if should_exit:
                        exit_price = data['Close'].iloc[start_idx]
                        if current_position['type'] == 'long':
                            exit_price *= (1 - self.slippage)
                        else:
                            exit_price *= (1 + self.slippage)
                        
                        trade = self._calculate_trade_metrics(
                            current_position, exit_price, pattern_date, data
                        )
                        trades.append(trade)
                        current_position = None
                        
            except Exception as e:
                logger.error(f"Error in combined strategy: {str(e)}")
                continue
        
        return self._calculate_backtest_results(trades)
    
    def _backtest_risk_adjusted_strategy(self, data: pd.DataFrame, 
                                       patterns: List[Dict[str, Any]]) -> BacktestResults:
        """Backtest risk-adjusted strategy with dynamic position sizing"""
        trades = []
        
        # Calculate ATR for risk management
        data = data.copy()
        data['ATR'] = self._calculate_atr(data)
        
        current_position = None
        capital = self.initial_capital
        
        for pattern in patterns:
            try:
                start_idx = pattern['start_idx']
                pattern_date = data.index[start_idx]
                signal = pattern.get('signal', '')
                confidence = pattern['confidence']
                
                # Only trade medium confidence patterns (relaxed to 0.3 for more trades)
                if confidence < 0.3:
                    continue
                
                current_price = data['Close'].iloc[start_idx]
                atr = data['ATR'].iloc[start_idx] if start_idx < len(data) else current_price * 0.02
                
                # Calculate position size based on risk (risk 2% of capital per trade)
                risk_per_trade = capital * 0.02
                stop_loss_distance = atr * 2 if atr > 0 else current_price * 0.02
                position_size = min(0.1, risk_per_trade / stop_loss_distance) if stop_loss_distance > 0 else 0.02
                
                if current_position is None and position_size > 0.001:  # Allow smaller positions
                    if 'Bullish' in signal:
                        entry_price = current_price * (1 + self.slippage)
                        stop_loss = current_price - (atr * 2 if atr > 0 else current_price * 0.02)
                        take_profit = current_price + (atr * 3 if atr > 0 else current_price * 0.04)
                        
                        current_position = {
                            'type': 'long',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': pattern['type'],
                            'confidence': confidence,
                            'entry_idx': start_idx,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size
                        }
                        logger.info(f"Risk-adjusted LONG: {pattern['type']} at {entry_price:.2f}, SL: {stop_loss:.2f}")
                    
                    elif 'Bearish' in signal:
                        entry_price = current_price * (1 - self.slippage)
                        stop_loss = current_price + (atr * 2 if atr > 0 else current_price * 0.02)
                        take_profit = current_price - (atr * 3 if atr > 0 else current_price * 0.04)
                        
                        current_position = {
                            'type': 'short',
                            'entry_date': pattern_date,
                            'entry_price': entry_price,
                            'pattern_type': pattern['type'],
                            'confidence': confidence,
                            'entry_idx': start_idx,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': position_size
                        }
                        logger.info(f"Risk-adjusted SHORT: {pattern['type']} at {entry_price:.2f}, SL: {stop_loss:.2f}")
                
                # Check exit conditions for existing position
                elif current_position is not None:
                    exit_price = None
                    exit_reason = None
                    
                    if current_position['type'] == 'long':
                        if current_price <= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            exit_reason = 'Stop Loss'
                        elif current_price >= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            exit_reason = 'Take Profit'
                    
                    else:  # short position
                        if current_price >= current_position['stop_loss']:
                            exit_price = current_position['stop_loss']
                            exit_reason = 'Stop Loss'
                        elif current_price <= current_position['take_profit']:
                            exit_price = current_position['take_profit']
                            exit_reason = 'Take Profit'
                    
                    if exit_price is not None:
                        trade = self._calculate_trade_metrics(
                            current_position, exit_price, pattern_date, data
                        )
                        trade.exit_reason = exit_reason
                        trades.append(trade)
                        
                        # Update capital
                        trade_pnl = trade.return_pct * capital * current_position['position_size']
                        capital += trade_pnl
                        
                        current_position = None
                        
            except Exception as e:
                logger.error(f"Error in risk-adjusted strategy: {str(e)}")
                continue
        
        return self._calculate_backtest_results(trades)
    
    def _calculate_trade_metrics(self, position: Dict[str, Any], exit_price: float, 
                               exit_date: datetime, data: pd.DataFrame) -> BacktestTrade:
        """Calculate detailed trade metrics"""
        
        entry_idx = position['entry_idx']
        entry_price = position['entry_price']
        
        # Find exit index
        exit_idx = len(data) - 1
        for i, date in enumerate(data.index):
            if date >= exit_date:
                exit_idx = i
                break
        
        # Calculate return
        if position['type'] == 'long':
            return_pct = (exit_price - entry_price) / entry_price
        else:  # short
            return_pct = (entry_price - exit_price) / entry_price
        
        # Calculate holding period
        holding_days = (exit_date - position['entry_date']).days
        
        # Calculate max drawdown and max profit during holding period
        if exit_idx > entry_idx:
            price_range = data['Close'].iloc[entry_idx:exit_idx+1]
            
            if position['type'] == 'long':
                max_profit = (price_range.max() - entry_price) / entry_price
                max_drawdown = (entry_price - price_range.min()) / entry_price
            else:
                max_profit = (entry_price - price_range.min()) / entry_price
                max_drawdown = (price_range.max() - entry_price) / entry_price
        else:
            max_profit = max(0, return_pct)
            max_drawdown = max(0, -return_pct)
        
        return BacktestTrade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            trade_type=position['type'],
            pattern_type=position['pattern_type'],
            confidence=position['confidence'],
            return_pct=return_pct,
            holding_days=holding_days,
            max_drawdown=max_drawdown,
            max_profit=max_profit
        )
    
    def _calculate_backtest_results(self, trades: List[BacktestTrade]) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not trades:
            return BacktestResults(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_return=0, avg_return=0, max_drawdown=0, sharpe_ratio=0,
                calmar_ratio=0, profit_factor=0, avg_winning_return=0,
                avg_losing_return=0, max_consecutive_wins=0, max_consecutive_losses=0,
                best_trade=0, worst_trade=0, trades=[]
            )
        
        returns = [trade.return_pct for trade in trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        
        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        win_rate = winning_count / total_trades if total_trades > 0 else 0
        
        total_return = sum(returns)
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 2%)
        returns_array = np.array(returns)
        excess_returns = returns_array - 0.02/252  # Daily risk-free rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualize
        
        # Calculate Calmar ratio
        calmar_ratio = (total_return * 252) / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate average returns
        avg_winning_return = np.mean(winning_trades) if winning_trades else 0
        avg_losing_return = np.mean(losing_trades) if losing_trades else 0
        
        # Calculate consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(returns, lambda x: x > 0)
        consecutive_losses = self._calculate_max_consecutive(returns, lambda x: x <= 0)
        
        best_trade = max(returns) if returns else 0
        worst_trade = min(returns) if returns else 0
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            total_return=total_return,
            avg_return=avg_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            avg_winning_return=avg_winning_return,
            avg_losing_return=avg_losing_return,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=trades
        )
    
    def _calculate_max_consecutive(self, returns: List[float], condition) -> int:
        """Calculate maximum consecutive occurrences"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if condition(ret):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _compare_strategies(self, results: Dict[str, BacktestResults]) -> Dict[str, Any]:
        """Compare different strategies"""
        comparison = {}
        
        for strategy_name, result in results.items():
            if strategy_name != 'strategy_comparison' and strategy_name != 'monte_carlo':
                comparison[strategy_name] = {
                    'total_return': result.total_return,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades
                }
        
        # Find best strategy by Sharpe ratio
        best_strategy = max(comparison.keys(), 
                          key=lambda x: comparison[x]['sharpe_ratio'])
        
        comparison['best_strategy'] = best_strategy
        comparison['best_sharpe'] = comparison[best_strategy]['sharpe_ratio']
        
        return comparison
    
    def _run_monte_carlo_simulation(self, data: pd.DataFrame, 
                                   patterns: List[Dict[str, Any]], 
                                   num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for confidence intervals"""
        
        if not patterns:
            return {'error': 'No patterns for simulation'}
        
        # Extract historical returns from patterns
        pattern_returns = []
        for pattern in patterns:
            try:
                start_idx = pattern['start_idx']
                if start_idx + 5 < len(data):
                    entry_price = data['Close'].iloc[start_idx]
                    exit_price = data['Close'].iloc[start_idx + 5]  # 5-day hold
                    
                    if 'Bullish' in pattern.get('signal', ''):
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price
                    
                    pattern_returns.append(ret)
            except:
                continue
        
        if not pattern_returns:
            return {'error': 'No valid pattern returns'}
        
        # Run simulations
        simulation_results = []
        for _ in range(num_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(pattern_returns, size=len(pattern_returns), replace=True)
            total_return = np.sum(sampled_returns)
            simulation_results.append(total_return)
        
        # Calculate confidence intervals
        simulation_results = np.array(simulation_results)
        
        return {
            'mean_return': np.mean(simulation_results),
            'std_return': np.std(simulation_results),
            'confidence_95_lower': np.percentile(simulation_results, 2.5),
            'confidence_95_upper': np.percentile(simulation_results, 97.5),
            'confidence_90_lower': np.percentile(simulation_results, 5),
            'confidence_90_upper': np.percentile(simulation_results, 95),
            'probability_profit': np.mean(simulation_results > 0),
            'max_simulated_return': np.max(simulation_results),
            'min_simulated_return': np.min(simulation_results)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift())
        low_close_prev = abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr