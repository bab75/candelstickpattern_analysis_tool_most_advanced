import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import logging
from scipy.stats import norm
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptionData:
    """Option contract data"""
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class StrategyResult:
    """Option strategy analysis result"""
    strategy_name: str
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    profit_probability: float
    expected_return: float
    risk_reward_ratio: float
    capital_required: float
    best_case_scenario: str
    worst_case_scenario: str

class OptionsAnalyzer:
    """Advanced options analysis and strategy evaluation"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
    def get_option_chain(self, symbol: str, expiry_date: str = None) -> Dict[str, Any]:
        """Get options chain data for a symbol"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return {'error': 'No options available for this symbol'}
            
            # Use specified expiry or next available
            if expiry_date and expiry_date in expirations:
                expiry = expiry_date
            else:
                expiry = expirations[0]  # Next expiration
            
            # Get option chain
            option_chain = ticker.option_chain(expiry)
            
            calls = option_chain.calls
            puts = option_chain.puts
            
            # Get current stock price
            stock_info = ticker.info
            current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
            
            # Calculate Greeks for options
            calls_with_greeks = self._add_greeks_to_options(calls, current_price, expiry, 'call')
            puts_with_greeks = self._add_greeks_to_options(puts, current_price, expiry, 'put')
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'expiry': expiry,
                'available_expiries': expirations,
                'calls': calls_with_greeks,
                'puts': puts_with_greeks,
                'total_call_volume': calls['volume'].sum(),
                'total_put_volume': puts['volume'].sum(),
                'put_call_ratio': puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {str(e)}")
            return {'error': f'Failed to get option chain: {str(e)}'}
    
    def analyze_option_strategies(self, symbol: str, current_price: float, 
                                calls: pd.DataFrame, puts: pd.DataFrame,
                                investment_amount: float = 10000) -> List[StrategyResult]:
        """Analyze various option strategies"""
        
        strategies = []
        
        try:
            # 1. Covered Call
            strategies.append(self._analyze_covered_call(symbol, current_price, calls, investment_amount))
            
            # 2. Cash-Secured Put
            strategies.append(self._analyze_cash_secured_put(symbol, current_price, puts, investment_amount))
            
            # 3. Long Call
            strategies.append(self._analyze_long_call(symbol, current_price, calls, investment_amount))
            
            # 4. Long Put
            strategies.append(self._analyze_long_put(symbol, current_price, puts, investment_amount))
            
            # 5. Bull Call Spread
            strategies.append(self._analyze_bull_call_spread(symbol, current_price, calls, investment_amount))
            
            # 6. Bear Put Spread
            strategies.append(self._analyze_bear_put_spread(symbol, current_price, puts, investment_amount))
            
            # 7. Iron Condor
            strategies.append(self._analyze_iron_condor(symbol, current_price, calls, puts, investment_amount))
            
            # 8. Straddle
            strategies.append(self._analyze_straddle(symbol, current_price, calls, puts, investment_amount))
            
            # 9. Strangle
            strategies.append(self._analyze_strangle(symbol, current_price, calls, puts, investment_amount))
            
            # Filter out None strategies and sort by expected return
            strategies = [s for s in strategies if s is not None]
            strategies.sort(key=lambda x: x.expected_return, reverse=True)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error analyzing option strategies: {str(e)}")
            return []
    
    def calculate_implied_volatility_rank(self, symbol: str, current_iv: float) -> Dict[str, Any]:
        """Calculate implied volatility rank and percentile"""
        
        try:
            # Get historical IV data (approximated using stock price volatility)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)  # 1 year
            
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                return {'error': 'No historical data available'}
            
            # Calculate historical volatility as proxy for IV
            returns = hist_data['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)  # 30-day annualized vol
            
            # Calculate IV rank and percentile
            iv_values = rolling_vol.dropna()
            
            if len(iv_values) == 0:
                return {'error': 'Insufficient data for IV analysis'}
            
            iv_rank = (current_iv - iv_values.min()) / (iv_values.max() - iv_values.min()) if iv_values.max() != iv_values.min() else 0.5
            iv_percentile = (iv_values < current_iv).sum() / len(iv_values)
            
            return {
                'current_iv': current_iv,
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'historical_iv_mean': iv_values.mean(),
                'historical_iv_std': iv_values.std(),
                'iv_assessment': self._assess_iv_level(iv_rank, iv_percentile),
                'trading_recommendation': self._get_iv_trading_recommendation(iv_rank, iv_percentile)
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV rank: {str(e)}")
            return {'error': f'Failed to calculate IV rank: {str(e)}'}
    
    def find_arbitrage_opportunities(self, calls: pd.DataFrame, puts: pd.DataFrame, 
                                   current_price: float, risk_free_rate: float = None) -> List[Dict[str, Any]]:
        """Find potential arbitrage opportunities"""
        
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        opportunities = []
        
        try:
            # Check for put-call parity violations
            for _, call_row in calls.iterrows():
                strike = call_row['strike']
                
                # Find corresponding put
                put_row = puts[puts['strike'] == strike]
                if put_row.empty:
                    continue
                
                put_row = put_row.iloc[0]
                
                # Put-call parity: C - P = S - K*e^(-r*T)
                # Where C = call price, P = put price, S = stock price, K = strike, r = risk-free rate, T = time to expiry
                
                call_price = (call_row['bid'] + call_row['ask']) / 2
                put_price = (put_row['bid'] + put_row['ask']) / 2
                
                # Calculate time to expiry (assuming 30 days for simplification)
                time_to_expiry = 30 / 365
                
                theoretical_diff = current_price - strike * np.exp(-risk_free_rate * time_to_expiry)
                actual_diff = call_price - put_price
                
                arbitrage_profit = abs(actual_diff - theoretical_diff)
                
                if arbitrage_profit > 0.50:  # Minimum $0.50 profit threshold
                    direction = 'Buy Call, Sell Put' if actual_diff < theoretical_diff else 'Sell Call, Buy Put'
                    
                    opportunities.append({
                        'type': 'Put-Call Parity Violation',
                        'strike': strike,
                        'call_price': call_price,
                        'put_price': put_price,
                        'theoretical_diff': theoretical_diff,
                        'actual_diff': actual_diff,
                        'arbitrage_profit': arbitrage_profit,
                        'direction': direction,
                        'confidence': 'High' if arbitrage_profit > 1.0 else 'Medium'
                    })
            
            # Sort by profit potential
            opportunities.sort(key=lambda x: x['arbitrage_profit'], reverse=True)
            
            return opportunities[:5]  # Return top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {str(e)}")
            return []
    
    def _add_greeks_to_options(self, options_df: pd.DataFrame, stock_price: float, 
                             expiry: str, option_type: str) -> pd.DataFrame:
        """Add Greeks calculations to options dataframe"""
        
        try:
            options_df = options_df.copy()
            
            # Calculate time to expiry
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            time_to_expiry = (expiry_date - datetime.now()).days / 365
            
            if time_to_expiry <= 0:
                time_to_expiry = 1/365  # Minimum 1 day
            
            greeks = []
            
            for _, row in options_df.iterrows():
                strike = row['strike']
                option_price = (row['bid'] + row['ask']) / 2 if row['ask'] > 0 else row['lastPrice']
                
                # Use implied volatility if available, otherwise estimate
                if 'impliedVolatility' in row and row['impliedVolatility'] > 0:
                    iv = row['impliedVolatility']
                else:
                    iv = self._estimate_implied_volatility(option_price, stock_price, strike, 
                                                         time_to_expiry, self.risk_free_rate, option_type)
                
                # Calculate Greeks using Black-Scholes
                delta, gamma, theta, vega, rho = self._calculate_greeks(
                    stock_price, strike, time_to_expiry, self.risk_free_rate, iv, option_type
                )
                
                greeks.append({
                    'calculated_iv': iv,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': rho
                })
            
            # Add Greeks to dataframe
            greeks_df = pd.DataFrame(greeks)
            result_df = pd.concat([options_df.reset_index(drop=True), greeks_df], axis=1)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding Greeks: {str(e)}")
            return options_df
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str) -> Tuple[float, float, float, float, float]:
        """Calculate option Greeks using Black-Scholes model"""
        
        try:
            if T <= 0 or sigma <= 0:
                return 0, 0, 0, 0, 0
            
            # Black-Scholes parameters
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                # Call option Greeks
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
                
            else:  # put option
                # Put option Greeks
                delta = -norm.cdf(-d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return delta, gamma, theta, vega, rho
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return 0, 0, 0, 0, 0
    
    def _estimate_implied_volatility(self, option_price: float, stock_price: float, 
                                   strike: float, time_to_expiry: float, 
                                   risk_free_rate: float, option_type: str) -> float:
        """Estimate implied volatility using Newton-Raphson method"""
        
        try:
            # Initial guess
            sigma = 0.3
            
            for _ in range(50):  # Maximum iterations
                # Black-Scholes price
                bs_price = self._black_scholes_price(stock_price, strike, time_to_expiry, 
                                                   risk_free_rate, sigma, option_type)
                
                # Vega (sensitivity to volatility)
                vega = self._calculate_vega(stock_price, strike, time_to_expiry, 
                                          risk_free_rate, sigma)
                
                if abs(vega) < 1e-6:
                    break
                
                # Newton-Raphson update
                sigma_new = sigma - (bs_price - option_price) / vega
                
                if abs(sigma_new - sigma) < 1e-6:
                    break
                
                sigma = max(0.01, min(5.0, sigma_new))  # Keep within reasonable bounds
            
            return sigma
            
        except Exception as e:
            logger.error(f"Error estimating IV: {str(e)}")
            return 0.3  # Default volatility
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)
            
        except Exception as e:
            logger.error(f"Error calculating BS price: {str(e)}")
            return 0
    
    def _calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega"""
        
        try:
            if T <= 0 or sigma <= 0:
                return 0
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            return vega
            
        except Exception as e:
            logger.error(f"Error calculating vega: {str(e)}")
            return 0
    
    def _analyze_covered_call(self, symbol: str, current_price: float, 
                            calls: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze covered call strategy"""
        
        try:
            # Find ATM or slightly OTM call
            atm_calls = calls[(calls['strike'] >= current_price) & (calls['strike'] <= current_price * 1.05)]
            
            if atm_calls.empty:
                return None
            
            best_call = atm_calls.iloc[0]
            strike = best_call['strike']
            premium = (best_call['bid'] + best_call['ask']) / 2
            
            # Calculate metrics
            shares_to_buy = int(investment_amount / current_price / 100) * 100  # Round to 100s
            contracts = shares_to_buy // 100
            
            if contracts == 0:
                return None
            
            premium_income = premium * contracts * 100
            stock_cost = shares_to_buy * current_price
            
            # Max profit: Premium + (Strike - Stock Price) if called away
            max_profit = premium_income + max(0, (strike - current_price) * shares_to_buy)
            
            # Max loss: Stock can go to zero minus premium received
            max_loss = stock_cost - premium_income
            
            # Breakeven: Stock price - premium per share
            breakeven = current_price - premium
            
            return StrategyResult(
                strategy_name="Covered Call",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.65,  # Estimated
                expected_return=max_profit / stock_cost if stock_cost > 0 else 0,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=stock_cost,
                best_case_scenario=f"Stock stays at/above ${strike:.2f} at expiration",
                worst_case_scenario=f"Stock drops significantly below ${breakeven:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing covered call: {str(e)}")
            return None
    
    def _analyze_cash_secured_put(self, symbol: str, current_price: float, 
                                puts: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze cash-secured put strategy"""
        
        try:
            # Find ATM or slightly OTM put
            atm_puts = puts[(puts['strike'] <= current_price) & (puts['strike'] >= current_price * 0.95)]
            
            if atm_puts.empty:
                return None
            
            best_put = atm_puts.iloc[0]
            strike = best_put['strike']
            premium = (best_put['bid'] + best_put['ask']) / 2
            
            # Calculate metrics
            contracts = int(investment_amount / (strike * 100))
            
            if contracts == 0:
                return None
            
            premium_income = premium * contracts * 100
            cash_secured = strike * contracts * 100
            
            # Max profit: Premium received
            max_profit = premium_income
            
            # Max loss: Strike price minus premium if stock goes to zero
            max_loss = cash_secured - premium_income
            
            # Breakeven: Strike - premium per share
            breakeven = strike - premium
            
            return StrategyResult(
                strategy_name="Cash-Secured Put",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.70,  # Estimated
                expected_return=max_profit / cash_secured if cash_secured > 0 else 0,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=cash_secured,
                best_case_scenario=f"Stock stays above ${strike:.2f} at expiration",
                worst_case_scenario=f"Stock drops significantly below ${breakeven:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing cash-secured put: {str(e)}")
            return None
    
    def _analyze_long_call(self, symbol: str, current_price: float, 
                         calls: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze long call strategy"""
        
        try:
            # Find ATM call
            atm_calls = calls[abs(calls['strike'] - current_price) == abs(calls['strike'] - current_price).min()]
            
            if atm_calls.empty:
                return None
            
            best_call = atm_calls.iloc[0]
            strike = best_call['strike']
            premium = (best_call['bid'] + best_call['ask']) / 2
            
            # Calculate metrics
            contracts = int(investment_amount / (premium * 100))
            
            if contracts == 0:
                return None
            
            total_premium = premium * contracts * 100
            
            # Max profit: Unlimited (theoretically)
            max_profit = float('inf')
            
            # Max loss: Premium paid
            max_loss = total_premium
            
            # Breakeven: Strike + premium per share
            breakeven = strike + premium
            
            return StrategyResult(
                strategy_name="Long Call",
                max_profit=10000,  # Use reasonable number instead of infinity
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.45,  # Estimated
                expected_return=1.0,  # Simplified
                risk_reward_ratio=10000 / max_loss if max_loss > 0 else 0,
                capital_required=total_premium,
                best_case_scenario=f"Stock rises significantly above ${breakeven:.2f}",
                worst_case_scenario=f"Stock stays below ${strike:.2f} at expiration"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing long call: {str(e)}")
            return None
    
    def _analyze_long_put(self, symbol: str, current_price: float, 
                        puts: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze long put strategy"""
        
        try:
            # Find ATM put
            atm_puts = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
            
            if atm_puts.empty:
                return None
            
            best_put = atm_puts.iloc[0]
            strike = best_put['strike']
            premium = (best_put['bid'] + best_put['ask']) / 2
            
            # Calculate metrics
            contracts = int(investment_amount / (premium * 100))
            
            if contracts == 0:
                return None
            
            total_premium = premium * contracts * 100
            
            # Max profit: Strike - premium (if stock goes to zero)
            max_profit = (strike - premium) * contracts * 100
            
            # Max loss: Premium paid
            max_loss = total_premium
            
            # Breakeven: Strike - premium per share
            breakeven = strike - premium
            
            return StrategyResult(
                strategy_name="Long Put",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.45,  # Estimated
                expected_return=max_profit / total_premium if total_premium > 0 else 0,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=total_premium,
                best_case_scenario=f"Stock drops significantly below ${breakeven:.2f}",
                worst_case_scenario=f"Stock stays above ${strike:.2f} at expiration"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing long put: {str(e)}")
            return None
    
    def _analyze_bull_call_spread(self, symbol: str, current_price: float, 
                                calls: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze bull call spread strategy"""
        
        try:
            # Find two calls with different strikes
            sorted_calls = calls.sort_values('strike')
            
            if len(sorted_calls) < 2:
                return None
            
            # Buy lower strike, sell higher strike
            long_call = sorted_calls.iloc[0]
            short_call = sorted_calls.iloc[1]
            
            long_strike = long_call['strike']
            short_strike = short_call['strike']
            long_premium = (long_call['bid'] + long_call['ask']) / 2
            short_premium = (short_call['bid'] + short_call['ask']) / 2
            
            net_premium = long_premium - short_premium
            
            if net_premium <= 0:
                return None
            
            # Calculate metrics
            contracts = int(investment_amount / (net_premium * 100))
            
            if contracts == 0:
                return None
            
            total_cost = net_premium * contracts * 100
            
            # Max profit: Difference in strikes - net premium paid
            max_profit = ((short_strike - long_strike) - net_premium) * contracts * 100
            
            # Max loss: Net premium paid
            max_loss = total_cost
            
            # Breakeven: Long strike + net premium
            breakeven = long_strike + net_premium
            
            return StrategyResult(
                strategy_name="Bull Call Spread",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.55,  # Estimated
                expected_return=max_profit / total_cost if total_cost > 0 else 0,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=total_cost,
                best_case_scenario=f"Stock rises above ${short_strike:.2f} at expiration",
                worst_case_scenario=f"Stock stays below ${long_strike:.2f} at expiration"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing bull call spread: {str(e)}")
            return None
    
    def _analyze_bear_put_spread(self, symbol: str, current_price: float, 
                               puts: pd.DataFrame, investment_amount: float) -> StrategyResult:
        """Analyze bear put spread strategy"""
        
        try:
            # Find two puts with different strikes
            sorted_puts = puts.sort_values('strike', ascending=False)
            
            if len(sorted_puts) < 2:
                return None
            
            # Buy higher strike, sell lower strike
            long_put = sorted_puts.iloc[0]
            short_put = sorted_puts.iloc[1]
            
            long_strike = long_put['strike']
            short_strike = short_put['strike']
            long_premium = (long_put['bid'] + long_put['ask']) / 2
            short_premium = (short_put['bid'] + short_put['ask']) / 2
            
            net_premium = long_premium - short_premium
            
            if net_premium <= 0:
                return None
            
            # Calculate metrics
            contracts = int(investment_amount / (net_premium * 100))
            
            if contracts == 0:
                return None
            
            total_cost = net_premium * contracts * 100
            
            # Max profit: Difference in strikes - net premium paid
            max_profit = ((long_strike - short_strike) - net_premium) * contracts * 100
            
            # Max loss: Net premium paid
            max_loss = total_cost
            
            # Breakeven: Long strike - net premium
            breakeven = long_strike - net_premium
            
            return StrategyResult(
                strategy_name="Bear Put Spread",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=0.55,  # Estimated
                expected_return=max_profit / total_cost if total_cost > 0 else 0,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=total_cost,
                best_case_scenario=f"Stock drops below ${short_strike:.2f} at expiration",
                worst_case_scenario=f"Stock stays above ${long_strike:.2f} at expiration"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing bear put spread: {str(e)}")
            return None
    
    def _analyze_iron_condor(self, symbol: str, current_price: float, 
                           calls: pd.DataFrame, puts: pd.DataFrame, 
                           investment_amount: float) -> StrategyResult:
        """Analyze iron condor strategy"""
        
        try:
            # This is a simplified iron condor analysis
            # In practice, you'd want to find optimal strike selection
            
            if len(calls) < 2 or len(puts) < 2:
                return None
            
            # Simplified: use first available options
            call_spread_cost = 100  # Placeholder
            put_spread_cost = 100   # Placeholder
            
            net_credit = 200  # Placeholder
            max_profit = net_credit
            max_loss = 1000  # Placeholder
            
            return StrategyResult(
                strategy_name="Iron Condor",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[current_price - 5, current_price + 5],  # Placeholder
                profit_probability=0.65,
                expected_return=0.15,
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=max_loss,
                best_case_scenario="Stock stays within expected range",
                worst_case_scenario="Stock moves significantly outside range"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing iron condor: {str(e)}")
            return None
    
    def _analyze_straddle(self, symbol: str, current_price: float, 
                        calls: pd.DataFrame, puts: pd.DataFrame, 
                        investment_amount: float) -> StrategyResult:
        """Analyze long straddle strategy"""
        
        try:
            # Find ATM call and put
            atm_call = calls[abs(calls['strike'] - current_price) == abs(calls['strike'] - current_price).min()]
            atm_put = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
            
            if atm_call.empty or atm_put.empty:
                return None
            
            call = atm_call.iloc[0]
            put = atm_put.iloc[0]
            
            call_premium = (call['bid'] + call['ask']) / 2
            put_premium = (put['bid'] + put['ask']) / 2
            total_premium = call_premium + put_premium
            
            # Calculate metrics
            contracts = int(investment_amount / (total_premium * 100))
            
            if contracts == 0:
                return None
            
            total_cost = total_premium * contracts * 100
            strike = call['strike']
            
            # Max profit: Unlimited (theoretically)
            max_profit = 10000  # Use reasonable number
            
            # Max loss: Total premium paid
            max_loss = total_cost
            
            # Breakeven points: Strike ± total premium
            breakeven_lower = strike - total_premium
            breakeven_upper = strike + total_premium
            
            return StrategyResult(
                strategy_name="Long Straddle",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_probability=0.40,  # Needs significant movement
                expected_return=1.0,  # Simplified
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=total_cost,
                best_case_scenario=f"Stock moves significantly beyond ${breakeven_lower:.2f} or ${breakeven_upper:.2f}",
                worst_case_scenario=f"Stock stays near ${strike:.2f} at expiration"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing straddle: {str(e)}")
            return None
    
    def _analyze_strangle(self, symbol: str, current_price: float, 
                        calls: pd.DataFrame, puts: pd.DataFrame, 
                        investment_amount: float) -> StrategyResult:
        """Analyze long strangle strategy"""
        
        try:
            # Find OTM call and put
            otm_calls = calls[calls['strike'] > current_price * 1.05]
            otm_puts = puts[puts['strike'] < current_price * 0.95]
            
            if otm_calls.empty or otm_puts.empty:
                return None
            
            call = otm_calls.iloc[0]
            put = otm_puts.iloc[-1]
            
            call_premium = (call['bid'] + call['ask']) / 2
            put_premium = (put['bid'] + put['ask']) / 2
            total_premium = call_premium + put_premium
            
            # Calculate metrics
            contracts = int(investment_amount / (total_premium * 100))
            
            if contracts == 0:
                return None
            
            total_cost = total_premium * contracts * 100
            
            # Max profit: Unlimited (theoretically)
            max_profit = 10000  # Use reasonable number
            
            # Max loss: Total premium paid
            max_loss = total_cost
            
            # Breakeven points
            call_strike = call['strike']
            put_strike = put['strike']
            breakeven_lower = put_strike - total_premium
            breakeven_upper = call_strike + total_premium
            
            return StrategyResult(
                strategy_name="Long Strangle",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                profit_probability=0.35,  # Needs significant movement
                expected_return=1.0,  # Simplified
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                capital_required=total_cost,
                best_case_scenario=f"Stock moves significantly beyond ${breakeven_lower:.2f} or ${breakeven_upper:.2f}",
                worst_case_scenario=f"Stock stays between ${put_strike:.2f} and ${call_strike:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing strangle: {str(e)}")
            return None
    
    def _assess_iv_level(self, iv_rank: float, iv_percentile: float) -> str:
        """Assess implied volatility level"""
        
        if iv_rank > 0.8 or iv_percentile > 0.8:
            return "Very High"
        elif iv_rank > 0.6 or iv_percentile > 0.6:
            return "High"
        elif iv_rank > 0.4 or iv_percentile > 0.4:
            return "Medium"
        elif iv_rank > 0.2 or iv_percentile > 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _get_iv_trading_recommendation(self, iv_rank: float, iv_percentile: float) -> str:
        """Get trading recommendation based on IV levels"""
        
        if iv_rank > 0.7:
            return "Consider selling options (high IV)"
        elif iv_rank < 0.3:
            return "Consider buying options (low IV)"
        else:
            return "Neutral IV environment"