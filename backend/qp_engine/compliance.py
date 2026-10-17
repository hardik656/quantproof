"""Compliance checks (mixin for QuantProofValidator)."""

import numpy as np

from qp_engine.core import (
    EPSILON,
    CheckResult,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_win_rate,
    convert_numpy,
)


class ComplianceChecks:
    def check_compliance_pass(self) -> CheckResult:
        gates = [
            self.returns.mean() > 0,
            calculate_sharpe(self.returns, self.trades_per_year) > 1.0,
            calculate_max_drawdown(self.returns) < 0.20,
            calculate_win_rate(self.returns) > 45,
            len(self.returns) > 50,
        ]
        passed = all(gates) if self.strict_mode else sum(gates) >= 4
        return CheckResult(
            "Prop Firm Compliance Gate (5-gate)",
            passed,
            100 if passed else 0,
            f"✅ PASSES ({sum(gates)}/5 gates)" if passed else f"❌ NEEDS FIXES ({sum(gates)}/5 gates)",
            "FTMO/Topstep reject 90% of strategies.",
            "Fix top 3 failing checks.",
            "Compliance",
        )

    def check_prop_firm_compliance(self) -> CheckResult:
        """Check prop firm compliance using the enhanced engine when available"""

        # If enhanced prop firm engine is available, use it
        if hasattr(self, "prop_firm_engine") and self.prop_firm_engine is not None:
            try:
                prop_firm_results = self.prop_firm_engine.validate_all_firms()
                # Convert NumPy types to native Python types for JSON serialization
                prop_firm_results = convert_numpy(prop_firm_results)
                print("✅ NUMPY CONVERSION APPLIED")

                # STEP 3 — VERIFY TYPES
                if prop_firm_results.get("summary") and "average_score" in prop_firm_results["summary"]:
                    print("TYPE CHECK:", type(prop_firm_results["summary"]["average_score"]))

                firms = prop_firm_results.get("firms", {})

                eligible = []
                violations = []

                for firm_name, firm_data in firms.items():
                    if firm_data.get("eligible", False):
                        eligible.append(firm_name)
                    else:
                        firm_violations = firm_data.get("violations", [])
                        if firm_violations:
                            violation_names = [v.get("name", "") for v in firm_violations[:2]]
                            violations.append(f"{firm_name}: {', '.join(violation_names)}")

                passed = len(eligible) >= 1

                if eligible:
                    value_str = f"Eligible: {', '.join(eligible)}"
                    insight_str = f"Meets {len(eligible)}/{len(firms)} prop firms."
                    fix_str = "Consider submitting to eligible firms."
                else:
                    value_str = "Not eligible"
                    insight_str = f"Violations: {' | '.join(violations[:2])}"
                    fix_str = "Reduce DD, hit targets."

                score = len(eligible) / len(firms) * 100
                return CheckResult(
                    "Prop Firm Compliance (5 Firms Analyzed)",
                    passed,
                    round(score, 1),
                    value_str,
                    insight_str,
                    fix_str,
                    "Compliance",
                )

            except Exception as e:
                print(f"Prop firm engine error: {e}")
                # Fall back to legacy method

        # Legacy fallback method (3 firms only)
        r = self.returns
        max_dd = calculate_max_drawdown(r)
        total_return = float(np.prod(1 + r) - 1)
        sharpe = calculate_sharpe(r, self.trades_per_year)
        firms = {
            "FTMO": {"max_dd": 0.10, "profit_target": 0.10},
            "Topstep": {"max_dd": 0.10, "profit_target": 0.06},
            "The5ers": {"max_dd": 0.10, "profit_target": 0.08},
        }
        eligible = []
        violations = []
        for name, rules in firms.items():
            dd_ok = max_dd <= rules["max_dd"]
            profit_ok = total_return >= rules["profit_target"]
            sharpe_ok = sharpe >= 1.0
            if dd_ok and profit_ok and sharpe_ok:
                eligible.append(name)
            else:
                v = []
                if not dd_ok:
                    v.append(f"DD {max_dd:.1%}>{rules['max_dd']:.0%}")
                if not profit_ok:
                    v.append(f"Return {total_return:.1%}<{rules['profit_target']:.0%}")
                if not sharpe_ok:
                    v.append(f"Sharpe {sharpe:.2f}<1.0")
                violations.append(f"{name}: {', '.join(v)}")
        passed = len(eligible) >= 1

        if eligible:
            value_str = f"Eligible: {', '.join(eligible)}"
            insight_str = f"Meets {len(eligible)}/3 prop firms."
            fix_str = "Consider submitting to eligible firms."
        else:
            value_str = "Not eligible"
            insight_str = f"Violations: {' | '.join(violations[:2])}"
            fix_str = "Reduce DD, hit targets."

        score = len(eligible) / 3 * 100
        return CheckResult(
            "Prop Firm Compliance (FTMO/Topstep/The5ers)",
            passed,
            round(score, 1),
            value_str,
            insight_str,
            fix_str,
            "Compliance",
        )

    def check_institutional_compliance(self) -> CheckResult:
        """
        INSTITUTIONAL FIX: separate institutional compliance from prop firm compliance.
        Prop firms test a 30-day challenge window with 10% DD limits — calibrated for
        retail traders, not institutional strategies.
        Institutional standard (CTA/fund): Sharpe>1.0, DD<25%, Calmar>1.0.
        This check replaces the compliance hard-cap that was incorrectly failing
        legitimate strategies with realistic drawdowns.
        """
        r = self.returns
        max_dd = calculate_max_drawdown(r)
        sharpe = calculate_sharpe(r, self.trades_per_year)
        n_years = max(len(r) / max(self.trades_per_year, 1), 0.1)
        cagr = float(np.prod(1 + r) ** (1.0 / n_years) - 1)
        calmar = cagr / (max_dd + EPSILON)

        criteria = {
            "Sharpe ≥ 1.0": sharpe >= 1.0,
            "MaxDD < 25%": max_dd < 0.25,
            "Calmar ≥ 1.0": calmar >= 1.0,
        }
        met = [k for k, v in criteria.items() if v]
        unmet = [k for k, v in criteria.items() if not v]
        score = len(met) / len(criteria) * 100
        passed = len(unmet) == 0
        detail = f"Sharpe={sharpe:.2f} | MaxDD={max_dd:.1%} | Calmar={calmar:.2f}"
        return CheckResult(
            "Institutional Compliance (CTA/Fund Standard)",
            passed,
            round(score, 1),
            f"{'✅ Meets institutional standard' if passed else f'⚠ Unmet: {unmet}'} | {detail}",
            "CTA/fund institutional standards: Sharpe≥1.0, MaxDD<25%, Calmar≥1.0. "
            "More appropriate than prop firm rules for multi-year backtests.",
            "For prop firm eligibility see the Prop Firm Compliance check.",
            "Compliance",
        )
