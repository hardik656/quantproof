"""
QuantProof Prop Firm Compliance Module
Python implementation for 5 major prop firms
Compatible with existing QuantProofValidator v6.2
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum

# Import QuantProofValidator at runtime to avoid circular import
def _get_validator():
    from validator import QuantProofValidator
    return QuantProofValidator

class PropFirm(Enum):
    FTMO = "FTMO"
    TOPSTEP = "Topstep"
    THE5ERS = "The5ers"
    BLUE_GUARDIAN = "BlueGuardian"
    EARN2TRADE = "Earn2Trade"

class PropFirmRules:
    """Configuration for prop firm compliance rules"""
    
    RULES = {
        PropFirm.FTMO: {
            "name": "FTMO",
            "max_drawdown": 0.10,        # 10% maximum drawdown
            "daily_loss": 0.05,          # 5% daily loss limit
            "profit_target": 0.10,       # 10% profit target
            "consistency_days": 30,       # 30 trading days
            "min_trades": 10,            # Minimum 10 trades
            "consistency_rule": "profit_target",  # FTMO consistency check
            "rules": [
                {
                    "name": "Max Drawdown",
                    "value": 0.10,
                    "type": "percentage",
                    "description": "Maximum peak-to-trough decline",
                    "critical": True
                },
                {
                    "name": "Daily Loss",
                    "value": 0.05,
                    "type": "percentage",
                    "description": "Maximum daily loss limit",
                    "critical": True
                },
                {
                    "name": "Profit Target",
                    "value": 0.10,
                    "type": "percentage",
                    "description": "Required profit target",
                    "critical": True
                },
                {
                    "name": "Consistency",
                    "value": 0.40,
                    "type": "ratio",
                    "description": "Daily profit consistency (40% of best day)",
                    "critical": False
                }
            ]
        },
        
        PropFirm.TOPSTEP: {
            "name": "Topstep",
            "max_loss": 0.04,             # 4% maximum loss
            "profit_target": 0.08,        # 8% profit target
            "consistency_ratio": 0.50,    # 50% consistency rule
            "trading_days": 20,           # 20 trading days
            "rules": [
                {
                    "name": "Max Loss",
                    "value": 0.04,
                    "type": "percentage",
                    "description": "Maximum total loss",
                    "critical": True
                },
                {
                    "name": "Profit Target",
                    "value": 0.08,
                    "type": "percentage",
                    "description": "Required profit target",
                    "critical": True
                },
                {
                    "name": "Consistency",
                    "value": 0.50,
                    "type": "ratio",
                    "description": "50% consistency rule",
                    "critical": True
                }
            ]
        },
        
        PropFirm.THE5ERS: {
            "name": "The5ers",
            "max_drawdown": 0.08,         # 8% maximum drawdown
            "profit_target": 0.10,        # 10% profit target
            "consistency_days": 4,        # 4-day consistency
            "rules": [
                {
                    "name": "Max Drawdown",
                    "value": 0.08,
                    "type": "percentage",
                    "description": "Maximum peak-to-trough decline",
                    "critical": True
                },
                {
                    "name": "Profit Target",
                    "value": 0.10,
                    "type": "percentage",
                    "description": "Required profit target",
                    "critical": True
                },
                {
                    "name": "Consistency",
                    "value": 0.25,
                    "type": "ratio",
                    "description": "4-day consistency (25% of best 4-day period)",
                    "critical": False
                }
            ]
        },
        
        PropFirm.BLUE_GUARDIAN: {
            "name": "Blue Guardian",
            "max_drawdown": 0.10,         # 10% maximum drawdown
            "daily_loss": 0.05,          # 5% daily loss
            "profit_target": 0.08,        # 8% profit target
            "rules": [
                {
                    "name": "Max Drawdown",
                    "value": 0.10,
                    "type": "percentage",
                    "description": "Maximum peak-to-trough decline",
                    "critical": True
                },
                {
                    "name": "Daily Loss",
                    "value": 0.05,
                    "type": "percentage",
                    "description": "Maximum daily loss limit",
                    "critical": True
                },
                {
                    "name": "Profit Target",
                    "value": 0.08,
                    "type": "percentage",
                    "description": "Required profit target",
                    "critical": True
                }
            ]
        },
        
        PropFirm.EARN2TRADE: {
            "name": "Earn2Trade",
            "max_drawdown": 0.12,         # 12% maximum drawdown
            "profit_target": 0.06,        # 6% profit target
            "consistency_days": 10,       # 10-day consistency
            "rules": [
                {
                    "name": "Max Drawdown",
                    "value": 0.12,
                    "type": "percentage",
                    "description": "Maximum peak-to-trough decline",
                    "critical": True
                },
                {
                    "name": "Profit Target",
                    "value": 0.06,
                    "type": "percentage",
                    "description": "Required profit target",
                    "critical": True
                },
                {
                    "name": "Consistency",
                    "value": 0.30,
                    "type": "ratio",
                    "description": "10-day consistency (30% of best 10-day period)",
                    "critical": False
                }
            ]
        }
    }
    
    @classmethod
    def get_rules(cls, firm: PropFirm) -> Dict:
        """Get rules for a specific prop firm"""
        return cls.RULES.get(firm, {})
    
    @classmethod
    def get_all_firms(cls) -> List[PropFirm]:
        """Get all supported prop firms"""
        return list(cls.RULES.keys())

class PropFirmComplianceEngine:
    """Prop firm compliance engine compatible with QuantProofValidator"""
    
    def __init__(self, validator):
        """
        Initialize with existing QuantProofValidator instance
        validator: QuantProofValidator instance with loaded data
        """
        # Get QuantProofValidator at runtime to avoid circular import
        QuantProofValidator = _get_validator()
        
        # Validate validator has required attributes
        if not hasattr(validator, 'returns'):
            raise ValueError("Validator must have 'returns' attribute. Ensure validator.run() has been called.")
        
        print(f"validator.returns exists: {hasattr(validator, 'returns')}")
        print(f"validator.returns shape: {len(validator.returns) if hasattr(validator, 'returns') else 'N/A'}")
        
        self.validator = validator
        self.rules_engine = PropFirmRules()
    
    def validate_all_firms(self) -> Dict:
        """Validate strategy against all prop firms"""
        results = {}
        
        for firm in self.rules_engine.get_all_firms():
            results[firm.value] = self.validate_firm(firm)
        
        return {
            "firms": results,
            "summary": self._generate_summary(results),
            "recommendations": self._generate_recommendations(results)
        }
    
    def validate_firm(self, firm: PropFirm) -> Dict:
        """Validate strategy against a specific prop firm"""
        rules = self.rules_engine.get_rules(firm)
        if not rules:
            raise ValueError(f"Unknown firm: {firm}")
        
        # Calculate metrics from existing validator
        metrics = self._calculate_metrics()
        
        results = {
            "firm": firm.value,
            "eligible": True,
            "score": 0,
            "violations": [],
            "passes": [],
            "metrics": metrics
        }
        
        # Check each rule
        for rule in rules["rules"]:
            rule_result = self._check_rule(metrics, rule)
            
            if rule_result["passed"]:
                results["passes"].append(rule_result)
            else:
                results["violations"].append(rule_result)
                if rule_result["critical"]:
                    results["eligible"] = False
        
        # Calculate compliance score
        results["score"] = self._calculate_compliance_score(results, len(rules["rules"]))
        
        return results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate trading metrics from validator data"""
        # Get equity curve from validator
        equity_curve = self._get_equity_curve()
        
        # Calculate basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            elif peak - value > max_drawdown:
                max_drawdown = peak - value
        
        max_drawdown_pct = max_drawdown / peak
        
        # Calculate daily P&L if dates available
        daily_pnl = self._calculate_daily_pnl()
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown_pct,
            "daily_losses": daily_pnl["losses"],
            "daily_profits": daily_pnl["profits"],
            "max_daily_loss": max(daily_pnl["losses"]) if daily_pnl["losses"] else 0,
            "best_daily_profit": max(daily_pnl["profits"]) if daily_pnl["profits"] else 0,
            "consistency_metrics": self._calculate_consistency_metrics(daily_pnl)
        }
    
    def _get_equity_curve(self) -> np.ndarray:
        """Get equity curve from validator"""
        # Use returns from validator
        returns = self.validator.returns
        
        # Start from 10000 (base equity)
        base_equity = 10000
        equity_curve = base_equity * np.cumprod(1 + returns)
        
        return equity_curve
    
    def _calculate_daily_pnl(self) -> Dict:
        """Calculate daily P&L from validator data"""
        if not hasattr(self.validator, 'df') or 'date' not in self.validator.df.columns:
            return {"profits": [], "losses": []}
        
        df = self.validator.df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Sum raw P&L per day
        daily_data = df.groupby(df['date'].dt.date)['pnl'].sum()
        profits = daily_data[daily_data > 0].tolist()
        losses = daily_data[daily_data < 0].abs().tolist()
        
        return {"profits": profits, "losses": losses}
    
    def _calculate_consistency_metrics(self, daily_pnl: Dict) -> Dict:
        """Calculate consistency metrics for different firms"""
        consistency = {
            "ftmo_consistency": 0,
            "topstep_consistency": 0,
            "the5ers_consistency": 0,
            "earn2trade_consistency": 0
        }
        
        profits = daily_pnl["profits"]
        
        # FTMO consistency (40% of best day)
        if profits and len(profits) > 0:
            best_day = max(profits)
            threshold = best_day * 0.40
            qualifying_days = len([p for p in profits if p >= threshold])
            consistency["ftmo_consistency"] = qualifying_days / len(profits)
        
        # Topstep consistency (50% rule)
        if profits and daily_pnl["losses"]:
            positive_days = len(profits)
            total_days = len(profits) + len(daily_pnl["losses"])
            consistency["topstep_consistency"] = positive_days / total_days if total_days > 0 else 0
        
        # The5ers consistency (4-day rolling)
        consistency["the5ers_consistency"] = self._calculate_rolling_consistency(profits, 4)
        
        # Earn2Trade consistency (10-day rolling)
        consistency["earn2trade_consistency"] = self._calculate_rolling_consistency(profits, 10)
        
        return consistency
    
    def _calculate_rolling_consistency(self, profits: List[float], window_size: int) -> float:
        """Calculate rolling consistency for given window size"""
        if len(profits) < window_size:
            return 0.0
        
        best_rolling_sum = 0
        for i in range(len(profits) - window_size + 1):
            rolling_sum = sum(profits[i:i + window_size])
            best_rolling_sum = max(best_rolling_sum, rolling_sum)
        
        threshold = best_rolling_sum * 0.25
        qualifying_windows = sum(1 for i in range(len(profits) - window_size + 1) 
                              if sum(profits[i:i + window_size]) >= threshold)
        
        total_windows = len(profits) - window_size + 1
        return qualifying_windows / total_windows if total_windows > 0 else 0
    
    def _check_rule(self, metrics: Dict, rule: Dict) -> Dict:
        """Check if a specific rule is passed"""
        rule_name = rule["name"]
        rule_value = rule["value"]
        
        if rule_name == "Max Drawdown":
            value = metrics["max_drawdown"]
            threshold = rule_value
            passed = value <= threshold
            
        elif rule_name == "Daily Loss":
            value = metrics["max_daily_loss"]
            threshold = rule_value
            passed = value <= threshold
            
        elif rule_name == "Max Loss":
            value = max(metrics["daily_losses"]) if metrics["daily_losses"] else 0
            threshold = rule_value
            passed = value <= threshold
            
        elif rule_name == "Profit Target":
            value = metrics["total_return"]
            threshold = rule_value
            passed = value >= threshold
            
        elif rule_name == "Consistency":
            # Determine which consistency metric to use based on rule value
            if rule_value == 0.40:  # FTMO
                consistency = metrics["consistency_metrics"]["ftmo_consistency"]
            elif rule_value == 0.50:  # Topstep
                consistency = metrics["consistency_metrics"]["topstep_consistency"]
            elif rule_value == 0.25:  # The5ers
                consistency = metrics["consistency_metrics"]["the5ers_consistency"]
            elif rule_value == 0.30:  # Earn2Trade
                consistency = metrics["consistency_metrics"]["earn2trade_consistency"]
            else:
                consistency = 0
            
            value = consistency
            threshold = rule_value
            passed = value >= threshold
            
        else:
            raise ValueError(f"Unknown rule: {rule_name}")
        
        return {
            "name": rule_name,
            "description": rule["description"],
            "value": float(value),
            "threshold": float(threshold),
            "passed": bool(passed),
            "critical": rule["critical"],
            "type": rule["type"],
            "actual": f"{(value * 100):.2f}%",
            "required": f"{(threshold * 100):.2f}%"
        }
    
    def _calculate_compliance_score(self, results: Dict, total_rules: int) -> int:
        """Calculate compliance score (0-100)"""
        passed_rules = len(results["passes"])
        critical_violations = len([v for v in results["violations"] if v["critical"]])
        
        # Base score from passed rules
        score = (passed_rules / total_rules) * 100
        
        # Heavy penalty for critical violations
        if critical_violations > 0:
            score = max(0, score - (critical_violations * 30))
        
        # Small penalty for non-critical violations
        non_critical_violations = len([v for v in results["violations"] if not v["critical"]])
        score = max(0, score - (non_critical_violations * 10))
        
        return round(score)
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary of all prop firm results"""
        firms = list(results.keys())
        eligible_firms = [firm for firm, result in results.items() if result["eligible"]]
        
        average_score = sum(result["score"] for result in results.values()) / len(results)
        
        return {
            "total_firms": len(firms),
            "eligible_firms": len(eligible_firms),
            "eligible_firm_names": eligible_firms,
            "average_score": round(average_score),
            "best_firm": eligible_firms[0] if eligible_firms else None
        }
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate recommendations based on violations"""
        recommendations = []
        all_violations = []
        
        # Collect all violations
        for firm, result in results.items():
            for violation in result["violations"]:
                all_violations.append({
                    **violation,
                    "firm": firm
                })
        
        # Group violations by type
        violations_by_type = {}
        for violation in all_violations:
            rule_name = violation["name"]
            if rule_name not in violations_by_type:
                violations_by_type[rule_name] = []
            violations_by_type[rule_name].append(violation)
        
        # Generate recommendations for each violation type
        for rule_name, violations in violations_by_type.items():
            recommendation = self._get_recommendation_for_violation(rule_name, violations[0])
            recommendation["affected_firms"] = [v["firm"] for v in violations]
            recommendation["severity"] = "high" if any(v["critical"] for v in violations) else "medium"
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_recommendation_for_violation(self, violation_name: str, violation: Dict) -> Dict:
        """Get recommendation for specific violation"""
        recommendations = {
            "Max Drawdown": {
                "title": "Reduce Maximum Drawdown",
                "description": f"Your current drawdown is {violation['actual']}. Target: {violation['required']}",
                "actions": [
                    "Implement tighter stop-losses",
                    "Reduce position sizing during volatile periods",
                    "Add trend-following filters",
                    "Improve risk management rules"
                ]
            },
            "Daily Loss": {
                "title": "Control Daily Loss Limits",
                "description": f"Your max daily loss is {violation['actual']}. Target: {violation['required']}",
                "actions": [
                    "Set daily loss limits",
                    "Reduce exposure after losses",
                    "Implement intraday risk controls",
                    "Use smaller position sizes"
                ]
            },
            "Profit Target": {
                "title": "Improve Profitability",
                "description": f"Your current return is {violation['actual']}. Target: {violation['required']}",
                "actions": [
                    "Optimize entry/exit signals",
                    "Improve win rate",
                    "Increase average trade size",
                    "Add profit-taking strategies"
                ]
            },
            "Consistency": {
                "title": "Improve Trading Consistency",
                "description": f"Your consistency is {violation['actual']}. Target: {violation['required']}",
                "actions": [
                    "Trade more consistently across days",
                    "Avoid over-trading on good days",
                    "Implement daily trading plan",
                    "Use position sizing rules"
                ]
            }
        }
        
        return recommendations.get(violation_name, {
            "title": "Address Trading Issue",
            "description": f"Review your trading strategy for {violation_name}",
            "actions": ["Analyze trading patterns", "Adjust strategy parameters"]
        })

# Integration helper for QuantProofValidator
class QuantProofValidatorWithPropFirms:
    """Extended QuantProofValidator with prop firm compliance"""
    
    def __init__(self, *args, **kwargs):
        # Get QuantProofValidator at runtime to avoid circular import
        QuantProofValidator = _get_validator()
        super().__init__(*args, **kwargs)
        self.prop_firm_engine = PropFirmComplianceEngine(self)
    
    def run_prop_firm_compliance(self) -> Dict:
        """Run prop firm compliance analysis"""
        return self.prop_firm_engine.validate_all_firms()
    
    def run_with_prop_firms(self) -> Dict:
        """Run standard validation plus prop firm compliance"""
        # Run standard validation
        standard_results = self.run()
        
        # Add prop firm compliance
        prop_firm_results = self.run_prop_firm_compliance()
        
        # Combine results
        combined_results = {
            **standard_results,
            "prop_firm_compliance": prop_firm_results,
            "enhanced": True
        }
        
        return combined_results



