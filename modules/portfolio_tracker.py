import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_date: datetime
    sector: str
    weight: float

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    beta: float
    alpha: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    var_95: float  # Value at Risk
    expected_shortfall: float

class PortfolioTracker:
    """Advanced portfolio tracking and analytics"""
    
    def __init__(self):
        self.positions = {}
        self.historical_data = {}
        self.benchmark_symbol = 'SPY'
        
    def add_position(self, symbol: str, quantity: float, avg_cost: float, 
                    entry_date: datetime = None) -> None:
        """Add or update position in portfolio"""
        
        if entry_date is None:
            entry_date = datetime.now()
            
        # Fetch current price and sector info
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', avg_cost))
            sector = info.get('sector', 'Unknown')
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            current_price = avg_cost
            sector = 'Unknown'
        
        market_value = quantity * current_price
        cost_basis = quantity * avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) if cost_basis != 0 else 0
        
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            entry_date=entry_date,
            sector=sector,
            weight=0  # Will be calculated in get_portfolio_metrics
        )
        
        logger.info(f"Added position: {symbol} - {quantity} shares at ${avg_cost:.2f}")
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from portfolio"""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Removed position: {symbol}")
    
    def update_positions(self) -> None:
        """Update all positions with current market prices"""
        
        symbols = list(self.positions.keys())
        if not symbols:
            return
        
        try:
            # Handle single vs multiple symbols for batch fetch
            if len(symbols) == 1:
                tickers = yf.download(symbols[0], period='1d', interval='1d', 
                                    auto_adjust=True, prepost=True, threads=True)
            else:
                tickers = yf.download(symbols, period='1d', interval='1d', 
                                    group_by='ticker', auto_adjust=True, 
                                    prepost=True, threads=True)
            
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        current_price = tickers['Close'].iloc[-1]
                    else:
                        current_price = tickers[symbol]['Close'].iloc[-1]
                    
                    position = self.positions[symbol]
                    position.current_price = float(current_price.iloc[0]) if hasattr(current_price, 'iloc') else float(current_price)
                    position.market_value = float(position.quantity * position.current_price)
                    
                    cost_basis = float(position.quantity * position.avg_cost)
                    position.unrealized_pnl = float(position.market_value - cost_basis)
                    position.unrealized_pnl_pct = float((position.unrealized_pnl / cost_basis) if cost_basis != 0 else 0)
                    
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in batch price update: {str(e)}")
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if not self.positions:
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Update positions first
        self.update_positions()
        
        # Calculate basic metrics
        total_value = float(sum(pos.market_value for pos in self.positions.values()))
        total_cost = float(sum(pos.quantity * pos.avg_cost for pos in self.positions.values()))
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) if total_cost != 0 else 0
        
        # Update position weights
        for position in self.positions.values():
            position.weight = float(position.market_value) / total_value if total_value != 0 else 0
        
        # Calculate advanced metrics
        try:
            beta, alpha = self._calculate_beta_alpha()
            sharpe_ratio = self._calculate_sharpe_ratio()
            volatility = self._calculate_volatility()
            max_drawdown = self._calculate_max_drawdown()
            var_95 = self._calculate_var(0.95)
            expected_shortfall = self._calculate_expected_shortfall(0.95)
            
            # Calculate daily P&L (simplified - would need historical tracking)
            daily_pnl = total_pnl * 0.01  # Placeholder
            daily_pnl_pct = daily_pnl / total_value if total_value != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            beta = alpha = sharpe_ratio = volatility = max_drawdown = 0
            var_95 = expected_shortfall = daily_pnl = daily_pnl_pct = 0
        
        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            beta=beta,
            alpha=alpha,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall
        )
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation by sector"""
        
        sector_values = {}
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        for position in self.positions.values():
            sector = position.sector
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position.market_value
        
        # Convert to percentages
        sector_allocation = {}
        for sector, value in sector_values.items():
            sector_allocation[sector] = (value / total_value) if total_value != 0 else 0
        
        return sector_allocation
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate detailed risk metrics"""
        
        symbols = list(self.positions.keys())
        if not symbols:
            return {}
        
        try:
            # Fetch historical data for correlation analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)  # 1 year
            
            data = yf.download(symbols, start=start_date, end=end_date, 
                             auto_adjust=True)['Close']
            
            if data.empty:
                return {}
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Portfolio returns (weighted)
            weights = np.array([self.positions[symbol].weight for symbol in symbols])
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            # Correlation matrix
            correlation_matrix = returns.corr()
            
            # Maximum single position risk
            max_position_risk = max(pos.weight for pos in self.positions.values()) * 100
            
            return {
                'portfolio_volatility': volatility,
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'correlation_matrix': correlation_matrix,
                'max_position_risk_pct': max_position_risk,
                'portfolio_beta': self._calculate_beta_alpha()[0],
                'diversification_ratio': len(symbols),
                'concentration_risk': 'High' if max_position_risk > 20 else 'Medium' if max_position_risk > 10 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def generate_rebalancing_suggestions(self, target_allocation: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate portfolio rebalancing suggestions"""
        
        suggestions = []
        current_allocation = self.get_sector_allocation()
        
        # Default target allocation if not provided
        if target_allocation is None:
            target_allocation = {
                'Technology': 0.30,
                'Healthcare': 0.15,
                'Financial Services': 0.15,
                'Consumer Cyclical': 0.15,
                'Industrials': 0.10,
                'Other': 0.15
            }
        
        total_value = sum(pos.market_value for pos in self.positions.values())
        
        for sector, target_pct in target_allocation.items():
            current_pct = current_allocation.get(sector, 0)
            difference = target_pct - current_pct
            
            if abs(difference) > 0.05:  # 5% threshold
                action = 'Increase' if difference > 0 else 'Decrease'
                amount = abs(difference) * total_value
                
                suggestions.append({
                    'sector': sector,
                    'action': action,
                    'current_allocation': current_pct,
                    'target_allocation': target_pct,
                    'difference_pct': difference,
                    'amount_to_trade': amount,
                    'priority': 'High' if abs(difference) > 0.10 else 'Medium'
                })
        
        # Sort by priority and difference size
        suggestions.sort(key=lambda x: (x['priority'] == 'High', abs(x['difference_pct'])), reverse=True)
        
        return suggestions
    
    def _calculate_beta_alpha(self) -> tuple:
        """Calculate portfolio beta and alpha vs benchmark"""
        
        symbols = list(self.positions.keys())
        if not symbols:
            return 0, 0
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            # Fetch portfolio and benchmark data
            benchmark_data = yf.download(self.benchmark_symbol, start=start_date, end=end_date)['Close']
            
            if len(symbols) == 1:
                # Single symbol case
                portfolio_data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if portfolio_data.empty or benchmark_data.empty:
                    return 0, 0
                portfolio_returns = portfolio_data.pct_change().dropna()
            else:
                # Multiple symbols case
                portfolio_data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if portfolio_data.empty or benchmark_data.empty:
                    return 0, 0
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (portfolio_data.pct_change().dropna() * weights).sum(axis=1)
            benchmark_returns = benchmark_data.pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns[common_dates]
            benchmark_returns = benchmark_returns[common_dates]
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Calculate alpha (annualized)
            portfolio_avg_return = portfolio_returns.mean() * 252
            benchmark_avg_return = benchmark_returns.mean() * 252
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            
            alpha = portfolio_avg_return - (risk_free_rate + beta * (benchmark_avg_return - risk_free_rate))
            
            return beta, alpha
            
        except Exception as e:
            logger.error(f"Error calculating beta/alpha: {str(e)}")
            return 0, 0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio"""
        
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return 0
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                portfolio_returns = data.pct_change().dropna()
            else:
                data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                returns = data.pct_change().dropna()
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (returns * weights).sum(axis=1)
            
            avg_return_val = portfolio_returns.mean() * 252  # Annualized
            volatility_val = portfolio_returns.std() * np.sqrt(252)  # Annualized
            avg_return = float(avg_return_val.iloc[0]) if hasattr(avg_return_val, 'iloc') else float(avg_return_val)
            volatility_val = float(volatility_val.iloc[0]) if hasattr(volatility_val, 'iloc') else float(volatility_val)
            risk_free_rate = 0.02  # 2%
            
            sharpe_ratio = (avg_return - risk_free_rate) / volatility_val if volatility_val != 0 else 0
            
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return 0
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                portfolio_returns = data.pct_change().dropna()
            else:
                data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                returns = data.pct_change().dropna()
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (returns * weights).sum(axis=1)
            
            volatility_val = portfolio_returns.std() * np.sqrt(252)  # Annualized
            volatility = float(volatility_val.iloc[0]) if hasattr(volatility_val, 'iloc') else float(volatility_val)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return 0
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                portfolio_returns = data.pct_change().dropna()
            else:
                data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                returns = data.pct_change().dropna()
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (returns * weights).sum(axis=1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            max_drawdown_val = drawdown.min()
            max_drawdown = float(max_drawdown_val.iloc[0]) if hasattr(max_drawdown_val, 'iloc') else float(max_drawdown_val)
            
            return abs(max_drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def _calculate_var(self, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return 0
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                portfolio_returns = data.pct_change().dropna()
            else:
                data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                returns = data.pct_change().dropna()
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (returns * weights).sum(axis=1)
            
            var = float(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0
    
    def _calculate_expected_shortfall(self, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        try:
            symbols = list(self.positions.keys())
            if not symbols:
                return 0
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)
            
            if len(symbols) == 1:
                data = yf.download(symbols[0], start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                portfolio_returns = data.pct_change().dropna()
            else:
                data = yf.download(symbols, start=start_date, end=end_date)['Close']
                if data.empty:
                    return 0
                returns = data.pct_change().dropna()
                weights = np.array([self.positions[symbol].weight for symbol in symbols])
                portfolio_returns = (returns * weights).sum(axis=1)
            
            var = float(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
            expected_shortfall_val = portfolio_returns[portfolio_returns <= var].mean()
            expected_shortfall = float(expected_shortfall_val.iloc[0]) if hasattr(expected_shortfall_val, 'iloc') else float(expected_shortfall_val)
            
            return abs(expected_shortfall)
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0