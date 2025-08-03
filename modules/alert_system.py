import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Trading alert data structure"""
    id: str
    symbol: str
    alert_type: str
    condition: str
    current_value: float
    target_value: float
    priority: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    created_at: datetime
    triggered_at: Optional[datetime]
    is_active: bool
    message: str
    action_recommended: str
    confidence: float

@dataclass
class AlertCondition:
    """Alert condition configuration"""
    condition_type: str  # 'price', 'volume', 'pattern', 'technical', 'volatility'
    operator: str  # '>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'
    value: float
    timeframe: str
    description: str

class AlertSystem:
    """Advanced real-time alert system for trading signals"""
    
    def __init__(self):
        self.alerts = {}
        self.triggered_alerts = []
        self.alert_conditions = {
            'price_breakout': self._check_price_breakout,
            'volume_spike': self._check_volume_spike,
            'pattern_formation': self._check_pattern_formation,
            'technical_signal': self._check_technical_signal,
            'volatility_change': self._check_volatility_change,
            'support_resistance': self._check_support_resistance,
            'momentum_shift': self._check_momentum_shift,
            'earnings_announcement': self._check_earnings_announcement,
            'options_activity': self._check_options_activity,
            'sector_rotation': self._check_sector_rotation
        }
        self.notification_methods = []
        
    def create_alert(self, symbol: str, alert_type: str, condition: AlertCondition,
                    priority: str = 'MEDIUM', message: str = None) -> str:
        """Create a new alert"""
        
        # Generate unique alert ID with microseconds to avoid duplicates
        alert_id = f"{symbol}_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        if message is None:
            message = f"{symbol} {alert_type} alert: {condition.description}"
        
        alert = Alert(
            id=alert_id,
            symbol=symbol,
            alert_type=alert_type,
            condition=condition.description,
            current_value=0.0,  # Will be updated when checking
            target_value=condition.value,
            priority=priority,
            created_at=datetime.now(),
            triggered_at=None,
            is_active=True,
            message=message,
            action_recommended="",
            confidence=0.0
        )
        
        self.alerts[alert_id] = {
            'alert': alert,
            'condition': condition
        }
        
        logger.info(f"Created alert {alert_id} for {symbol}")
        return alert_id
    
    def check_alerts(self, data_fetcher, pattern_recognition, advanced_analytics) -> List[Alert]:
        """Check all active alerts and return triggered ones"""
        
        triggered_alerts = []
        
        for alert_id, alert_data in self.alerts.items():
            alert = alert_data['alert']
            condition = alert_data['condition']
            
            if not alert.is_active:
                continue
            
            try:
                # Check if condition is met
                is_triggered, current_value, confidence, action = self._evaluate_condition(
                    alert.symbol, condition, data_fetcher, pattern_recognition, advanced_analytics
                )
                
                # Update current value
                alert.current_value = current_value
                alert.confidence = confidence
                alert.action_recommended = action
                
                if is_triggered and alert.triggered_at is None:
                    # Alert triggered for the first time
                    alert.triggered_at = datetime.now()
                    alert.is_active = False  # Deactivate after triggering
                    
                    triggered_alerts.append(alert)
                    self.triggered_alerts.append(alert)
                    
                    logger.info(f"Alert triggered: {alert_id}")
                    
                    # Send notifications
                    self._send_notifications(alert)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert_id}: {str(e)}")
                continue
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert_data['alert'] for alert_data in self.alerts.values() 
                if alert_data['alert'].is_active]
    
    def get_triggered_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts triggered in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [alert for alert in self.triggered_alerts 
                if alert.triggered_at and alert.triggered_at >= cutoff_time]
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert {alert_id}")
            return True
        return False
    
    def create_price_alert(self, symbol: str, target_price: float, 
                          direction: str = 'above') -> str:
        """Create a simple price alert"""
        
        operator = '>=' if direction == 'above' else '<='
        condition = AlertCondition(
            condition_type='price',
            operator=operator,
            value=target_price,
            timeframe='1min',
            description=f"Price {direction} ${target_price:.2f}"
        )
        
        priority = 'HIGH' if abs(target_price) > 1000 else 'MEDIUM'
        
        return self.create_alert(
            symbol=symbol,
            alert_type='price_breakout',
            condition=condition,
            priority=priority,
            message=f"{symbol} price alert: Target ${target_price:.2f} {direction}"
        )
    
    def create_volume_alert(self, symbol: str, volume_multiplier: float = 2.0) -> str:
        """Create volume spike alert"""
        
        condition = AlertCondition(
            condition_type='volume',
            operator='>=',
            value=volume_multiplier,
            timeframe='1h',
            description=f"Volume {volume_multiplier}x above average"
        )
        
        return self.create_alert(
            symbol=symbol,
            alert_type='volume_spike',
            condition=condition,
            priority='HIGH',
            message=f"{symbol} volume spike: {volume_multiplier}x normal volume"
        )
    
    def create_pattern_alert(self, symbol: str, pattern_types: List[str]) -> str:
        """Create candlestick pattern alert"""
        
        condition = AlertCondition(
            condition_type='pattern',
            operator='==',
            value=1.0,  # Pattern detected
            timeframe='1d',
            description=f"Pattern formation: {', '.join(pattern_types)}"
        )
        
        return self.create_alert(
            symbol=symbol,
            alert_type='pattern_formation',
            condition=condition,
            priority='MEDIUM',
            message=f"{symbol} pattern alert: {', '.join(pattern_types)} detected"
        )
    
    def create_technical_alert(self, symbol: str, indicator: str, 
                             value: float, operator: str = '>=') -> str:
        """Create technical indicator alert"""
        
        condition = AlertCondition(
            condition_type='technical',
            operator=operator,
            value=value,
            timeframe='1h',
            description=f"{indicator} {operator} {value}"
        )
        
        return self.create_alert(
            symbol=symbol,
            alert_type='technical_signal',
            condition=condition,
            priority='MEDIUM',
            message=f"{symbol} technical alert: {indicator} {operator} {value}"
        )
    
    def create_smart_alerts_for_symbol(self, symbol: str, data_fetcher, 
                                     pattern_recognition, advanced_analytics) -> List[str]:
        """Create intelligent alerts based on current market conditions"""
        
        alert_ids = []
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1d')
            
            if data is None or data.empty:
                return alert_ids
            
            current_price = data['Close'].iloc[-1]
            
            # 1. Support/Resistance breakout alerts
            levels = advanced_analytics.calculate_support_resistance_levels(data)
            
            for support in levels.get('support', []):
                if abs(current_price - support) / current_price < 0.05:  # Within 5%
                    alert_id = self.create_price_alert(symbol, support * 0.995, 'below')
                    alert_ids.append(alert_id)
            
            for resistance in levels.get('resistance', []):
                if abs(current_price - resistance) / current_price < 0.05:  # Within 5%
                    alert_id = self.create_price_alert(symbol, resistance * 1.005, 'above')
                    alert_ids.append(alert_id)
            
            # 2. Technical indicator alerts
            rsi = advanced_analytics.calculate_rsi(data)
            current_rsi = rsi.iloc[-1]
            
            if current_rsi > 60:
                alert_id = self.create_technical_alert(symbol, 'RSI', 70, '>=')
                alert_ids.append(alert_id)
            elif current_rsi < 40:
                alert_id = self.create_technical_alert(symbol, 'RSI', 30, '<=')
                alert_ids.append(alert_id)
            
            # 3. Volume spike alert
            alert_id = self.create_volume_alert(symbol, 2.0)
            alert_ids.append(alert_id)
            
            # 4. Pattern formation alert
            alert_id = self.create_pattern_alert(symbol, ['Hammer', 'Doji', 'Engulfing'])
            alert_ids.append(alert_id)
            
            logger.info(f"Created {len(alert_ids)} smart alerts for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating smart alerts for {symbol}: {str(e)}")
        
        return alert_ids
    
    def _evaluate_condition(self, symbol: str, condition: AlertCondition, 
                          data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Evaluate if alert condition is met"""
        
        condition_type = condition.condition_type
        
        if condition_type in self.alert_conditions:
            return self.alert_conditions[condition_type](
                symbol, condition, data_fetcher, pattern_recognition, advanced_analytics
            )
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return False, 0.0, 0.0, ""
    
    def _check_price_breakout(self, symbol: str, condition: AlertCondition,
                            data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check price breakout condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=1)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1min')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            current_price = data['Close'].iloc[-1]
            target_price = condition.value
            
            is_triggered = False
            confidence = 0.8
            action = ""
            
            if condition.operator == '>=' and current_price >= target_price:
                is_triggered = True
                action = f"BUY signal - Price broke above ${target_price:.2f}"
            elif condition.operator == '<=' and current_price <= target_price:
                is_triggered = True
                action = f"SELL signal - Price broke below ${target_price:.2f}"
            
            return is_triggered, current_price, confidence, action
            
        except Exception as e:
            logger.error(f"Error checking price breakout: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_volume_spike(self, symbol: str, condition: AlertCondition,
                          data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check volume spike condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1h')
            
            if data is None or data.empty or 'Volume' not in data.columns:
                return False, 0.0, 0.0, ""
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            target_ratio = condition.value
            
            is_triggered = volume_ratio >= target_ratio
            confidence = min(0.9, volume_ratio / target_ratio) if target_ratio > 0 else 0
            
            action = f"High volume activity - {volume_ratio:.1f}x normal" if is_triggered else ""
            
            return is_triggered, volume_ratio, confidence, action
            
        except Exception as e:
            logger.error(f"Error checking volume spike: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_pattern_formation(self, symbol: str, condition: AlertCondition,
                               data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check pattern formation condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1d')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            # Analyze patterns
            patterns = pattern_recognition.analyze_patterns(data, pattern_recognition.patterns.keys())
            
            # Check for recent patterns (last 2 days)
            recent_patterns = [p for p in patterns if p['date'] >= end_date - timedelta(days=2)]
            
            is_triggered = len(recent_patterns) > 0
            confidence = max([p['confidence'] for p in recent_patterns]) if recent_patterns else 0
            
            if is_triggered:
                pattern_names = [p['type'] for p in recent_patterns]
                signals = [p.get('signal', 'Neutral') for p in recent_patterns]
                action = f"Pattern detected: {', '.join(pattern_names)} - {', '.join(signals)}"
            else:
                action = ""
            
            return is_triggered, len(recent_patterns), confidence, action
            
        except Exception as e:
            logger.error(f"Error checking pattern formation: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_technical_signal(self, symbol: str, condition: AlertCondition,
                              data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check technical indicator condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1h')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            # Calculate RSI as example (can be extended for other indicators)
            rsi = advanced_analytics.calculate_rsi(data)
            current_rsi = rsi.iloc[-1]
            
            target_value = condition.value
            is_triggered = False
            confidence = 0.7
            action = ""
            
            if condition.operator == '>=' and current_rsi >= target_value:
                is_triggered = True
                action = f"RSI overbought: {current_rsi:.1f} >= {target_value}"
            elif condition.operator == '<=' and current_rsi <= target_value:
                is_triggered = True
                action = f"RSI oversold: {current_rsi:.1f} <= {target_value}"
            
            return is_triggered, current_rsi, confidence, action
            
        except Exception as e:
            logger.error(f"Error checking technical signal: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_volatility_change(self, symbol: str, condition: AlertCondition,
                               data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check volatility change condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1d')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            # Calculate volatility
            returns = data['Close'].pct_change().dropna()
            current_vol = returns.rolling(5).std().iloc[-1] * np.sqrt(252)  # 5-day annualized vol
            avg_vol = returns.rolling(20).std().mean() * np.sqrt(252)  # 20-day average
            
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            target_ratio = condition.value
            
            is_triggered = vol_ratio >= target_ratio
            confidence = min(0.8, vol_ratio / target_ratio) if target_ratio > 0 else 0
            
            action = f"Volatility spike: {vol_ratio:.1f}x normal" if is_triggered else ""
            
            return is_triggered, vol_ratio, confidence, action
            
        except Exception as e:
            logger.error(f"Error checking volatility change: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_support_resistance(self, symbol: str, condition: AlertCondition,
                                data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check support/resistance level condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1d')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            current_price = data['Close'].iloc[-1]
            levels = advanced_analytics.calculate_support_resistance_levels(data)
            
            # Check if price is near support/resistance
            all_levels = levels.get('support', []) + levels.get('resistance', [])
            
            for level in all_levels:
                if abs(current_price - level) / current_price < 0.02:  # Within 2%
                    is_triggered = True
                    confidence = 0.75
                    level_type = 'Support' if level in levels.get('support', []) else 'Resistance'
                    action = f"Price near {level_type} level: ${level:.2f}"
                    return is_triggered, level, confidence, action
            
            return False, current_price, 0.0, ""
            
        except Exception as e:
            logger.error(f"Error checking support/resistance: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_momentum_shift(self, symbol: str, condition: AlertCondition,
                            data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check momentum shift condition"""
        
        try:
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)
            data = data_fetcher.fetch_data(symbol, start_date, end_date, '1d')
            
            if data is None or data.empty:
                return False, 0.0, 0.0, ""
            
            # Calculate MACD
            macd_data = advanced_analytics.calculate_macd(data)
            
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            prev_macd = macd_data['macd'].iloc[-2]
            prev_signal = macd_data['signal'].iloc[-2]
            
            # Check for MACD crossover
            is_triggered = False
            confidence = 0.6
            action = ""
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                is_triggered = True
                action = "MACD bullish crossover detected"
            elif current_macd < current_signal and prev_macd >= prev_signal:
                is_triggered = True
                action = "MACD bearish crossover detected"
            
            return is_triggered, current_macd - current_signal, confidence, action
            
        except Exception as e:
            logger.error(f"Error checking momentum shift: {str(e)}")
            return False, 0.0, 0.0, ""
    
    def _check_earnings_announcement(self, symbol: str, condition: AlertCondition,
                                   data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check earnings announcement condition"""
        
        # This would require earnings calendar data
        # For now, return placeholder
        return False, 0.0, 0.0, ""
    
    def _check_options_activity(self, symbol: str, condition: AlertCondition,
                              data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check unusual options activity condition"""
        
        # This would require options flow data
        # For now, return placeholder
        return False, 0.0, 0.0, ""
    
    def _check_sector_rotation(self, symbol: str, condition: AlertCondition,
                             data_fetcher, pattern_recognition, advanced_analytics) -> tuple:
        """Check sector rotation condition"""
        
        # This would require sector performance data
        # For now, return placeholder
        return False, 0.0, 0.0, ""
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications"""
        
        # For now, just log the alert
        # In production, this would send email, SMS, push notifications, etc.
        
        logger.info(f"ALERT TRIGGERED: {alert.message}")
        logger.info(f"Priority: {alert.priority}")
        logger.info(f"Action: {alert.action_recommended}")
        logger.info(f"Confidence: {alert.confidence:.1%}")
        
        # Add to notification queue or send via external services
        
    def add_notification_method(self, method: Callable) -> None:
        """Add a notification method"""
        self.notification_methods.append(method)
    
    def export_alerts_to_json(self, filepath: str) -> None:
        """Export alerts to JSON file"""
        
        try:
            alert_data = {}
            
            for alert_id, data in self.alerts.items():
                alert = data['alert']
                condition = data['condition']
                
                alert_data[alert_id] = {
                    'alert': asdict(alert),
                    'condition': asdict(condition)
                }
            
            # Convert datetime objects to strings
            for alert_id in alert_data:
                alert_info = alert_data[alert_id]['alert']
                alert_info['created_at'] = alert_info['created_at'].isoformat() if alert_info['created_at'] else None
                alert_info['triggered_at'] = alert_info['triggered_at'].isoformat() if alert_info['triggered_at'] else None
            
            with open(filepath, 'w') as f:
                json.dump(alert_data, f, indent=2)
                
            logger.info(f"Exported {len(alert_data)} alerts to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting alerts: {str(e)}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        triggered_alerts = len(self.triggered_alerts)
        
        # Count by priority
        priority_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for alert_data in self.alerts.values():
            priority = alert_data['alert'].priority
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        # Count by type
        type_counts = {}
        for alert_data in self.alerts.values():
            alert_type = alert_data['alert'].alert_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'triggered_alerts': triggered_alerts,
            'priority_distribution': priority_counts,
            'type_distribution': type_counts,
            'average_confidence': np.mean([alert_data['alert'].confidence 
                                         for alert_data in self.alerts.values()]),
            'last_update': datetime.now().isoformat()
        }