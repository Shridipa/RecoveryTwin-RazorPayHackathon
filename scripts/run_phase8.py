"""
Phase 8 — Financial Policy Simulation & Stress Testing

Combines all previous phases into a comprehensive financial evaluation:
  - Baseline policy comparison
  - Named scenario evaluation
  - Cost sensitivity analysis
  - Treatment degradation analysis
  - Time discount sensitivity
  - Retry limit sensitivity
  - Monte Carlo simulation
  - Break-even analysis
  - Robustness scoring
  - Segment analysis
  - Leakage audit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from datetime import datetime

from recoverytwin.financial.economic import (
    FinancialEvaluator, load_financial_config,
    compute_expected_revenue, compute_time_discount,
)
from recoverytwin.financial.scenario import (
    ScenarioRunner, load_scenarios, compute_robustness_score, compute_worst_case,
)
from recoverytwin.financial.monte_carlo import MonteCarloSimulator, load_mc_config
from recoverytwin.financial.sensitivity import (
    cost_sensitivity, degradation_sensitivity, time_discount_sensitivity,
    retry_limit_sensitivity, find_breakeven_cost, segment_analysis,
)
from recoverytwin.financial.leakage import audit_financial_leakage
from recoverytwin.financial.report import generate_phase8_report
from recoverytwin.decision.engine import (
    DecisionEngine, load_policy_config, ACTION_NAMES, ACTION_LABELS, DEFAULT_COSTS,
)
from recoverytwin.causal.s_learner import SLearner


def main():
    print("=" * 60)
    print("RECOVERYTWIN — PHASE 8")
    print("FINANCIAL POLICY SIMULATION & STRESS TESTING")
    print("=" * 60)
    print()

    steps_total = 12
    step = 0

    # ================================================================
    # [1] Load data
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Loading validated data ...", end=" ")
    data_dir = Path("data/processed/debug")
    if not (data_dir / "train.parquet").exists():
        print("FAIL — data not found. Run generate_and_validate.py first.")
        sys.exit(1)

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    print(f"PASS (train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")

    # ================================================================
    # [2] Train S-Learner
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Training S-Learner ...", end=" ")
    s_learner = SLearner(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_samples=50, random_state=42
    )
    s_learner.fit(train_df)
    print("PASS")

    # ================================================================
    # [3] Load policy config & build decision table
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Building decision table ...", end=" ")
    config = load_policy_config()
    engine = DecisionEngine(config)

    def predict_fn(df, action):
        return s_learner.predict_counterfactual(df, treatment=action)

    decision_table = engine.build_decision_table(test_df, predict_fn, survival_probs=None)
    print(f"PASS ({len(decision_table)} payments)")

    # ================================================================
    # [4] Baseline policy evaluation
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Evaluating baseline policies ...", end=" ")
    fin_config = load_financial_config()
    evaluator = FinancialEvaluator(
        intervention_costs=fin_config.get("intervention_costs", DEFAULT_COSTS),
    )
    baseline_policies = evaluator.evaluate_comparison_policies(
        test_df, decision_table, action_costs=DEFAULT_COSTS,
    )
    baseline_summary = {
        name: {
            "net_revenue": pol["net_revenue"],
            "recovery_rate": pol["recovery_rate"],
            "n_interventions": pol["n_interventions"],
            "total_cost": pol["total_cost"],
        }
        for name, pol in baseline_policies.items()
    }
    print("PASS")

    # ================================================================
    # [5] Named scenario evaluation
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running named scenarios ...", end=" ")
    scenario_runner = ScenarioRunner(test_df, decision_table, evaluator)
    scenario_results = scenario_runner.run_all_scenarios()
    print(f"PASS ({len(scenario_results)} scenarios)")

    # ================================================================
    # [6] Cost sensitivity
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running cost sensitivity ...", end=" ")
    cost_results = cost_sensitivity(test_df, decision_table, evaluator)
    print(f"PASS ({len(cost_results)} cost levels)")

    # ================================================================
    # [7] Degradation sensitivity
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running degradation sensitivity ...", end=" ")
    deg_results = degradation_sensitivity(test_df, decision_table, evaluator)
    print(f"PASS ({len(deg_results)} degradation levels)")

    # ================================================================
    # [8] Time discount sensitivity
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running time discount sensitivity ...", end=" ")
    td_results = time_discount_sensitivity(test_df, decision_table, evaluator)
    print(f"PASS ({len(td_results)} lambda values)")

    # ================================================================
    # [9] Retry limit sensitivity
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running retry limit sensitivity ...", end=" ")
    rl_results = retry_limit_sensitivity(
        test_df, decision_table, engine.policy_filter,
    )
    print(f"PASS ({len(rl_results)} retry limits)")

    # ================================================================
    # [10] Monte Carlo
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running Monte Carlo simulation ...", end=" ")
    mc_config = load_mc_config()
    mc_sim = MonteCarloSimulator(
        n_simulations=mc_config.get("n_simulations", 500),
        random_seed=mc_config.get("random_seed", 42),
    )

    # Get true potential outcomes for MC
    true_po = np.column_stack([
        test_df[f"potential_outcome_{a}"].values.astype(float)
        for a in range(4)
    ])

    mc_results = mc_sim.compare_policies(
        test_df, decision_table, true_po,
        action_costs=DEFAULT_COSTS,
        amount_noise_std=0.05,
    )
    print(f"PASS ({mc_config.get('n_simulations', 500)} simulations)")

    # ================================================================
    # [11] Break-even analysis
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Running break-even analysis ...", end=" ")
    breakeven = []
    for action in [1, 2, 3]:
        be = find_breakeven_cost(test_df, decision_table, evaluator, action=action)
        breakeven.append(be)
    print("PASS")

    # ================================================================
    # [12] Robustness + Worst Case + Segments + Leakage
    # ================================================================
    step += 1
    print(f"[{step}/{steps_total}] Computing robustness & segments ...", end=" ")

    robustness = compute_robustness_score(scenario_results)
    worst_case = compute_worst_case(scenario_results)

    # Segment analyses
    seg_failure = segment_analysis(test_df, decision_table, evaluator, "failure_reason")
    seg_amount = segment_analysis(
        test_df.assign(amount_bucket=pd.cut(
            test_df["amount"],
            bins=[0, 500, 2000, 5000, 500000],
            labels=["low", "medium", "high", "very_high"],
        )),
        decision_table, evaluator, "amount_bucket",
    )
    segment_results = seg_failure + seg_amount

    # Leakage audit
    leakage = audit_financial_leakage(test_df, decision_table)
    print("PASS")

    # ================================================================
    # SAVE REPORTS
    # ================================================================
    report = generate_phase8_report(
        baseline_policies=baseline_summary,
        scenario_results=scenario_results,
        cost_sensitivity=cost_results,
        degradation_sensitivity=deg_results,
        time_discount_sensitivity=td_results,
        retry_limit_sensitivity=rl_results,
        monte_carlo_results=mc_results,
        breakeven_results=breakeven,
        robustness=robustness,
        worst_case=worst_case,
        segment_results=segment_results,
        leakage_audit=leakage,
    )

    # ================================================================
    # PRINT RESULTS
    # ================================================================
    print()
    print("=" * 60)
    print("POLICY COMPARISON (Baseline)")
    print("=" * 60)

    dn_val = baseline_policies["do_nothing"]["net_revenue"]
    print(f"{'Policy':<25} {'Net Revenue':>15} {'vs Do Nothing':>15}")
    print("-" * 58)
    for name, pol in baseline_summary.items():
        val = pol["net_revenue"]
        delta = val - dn_val
        label = name.replace("_", " ").title()
        print(f"{label:<25} Rs.{val:>12,.0f} {delta:>+15,.0f}")

    print()
    print("=" * 60)
    print("ACTION ALLOCATION")
    print("=" * 60)

    if "recommended_action" in decision_table.columns:
        rt_actions = decision_table["recommended_action"].values
        for a in range(4):
            pct = np.mean(rt_actions == a) * 100
            print(f"  {ACTION_LABELS[a]:<25} {pct:>5.1f}%")
        print(f"  Intervention rate: {(rt_actions > 0).mean():.1%}")

    print()
    print("=" * 60)
    print("MONTE CARLO SUMMARY")
    print("=" * 60)

    for pol_name in ["do_nothing", "recoverytwin", "max_probability", "oracle"]:
        if pol_name in mc_results:
            mc = mc_results[pol_name]
            print(f"  {pol_name.replace('_', ' ').title():<25}")
            print(f"    Mean:  Rs.{mc['mean_net_revenue']:>12,.0f}")
            print(f"    P5:    Rs.{mc['p5']:>12,.0f}")
            print(f"    P95:   Rs.{mc['p95']:>12,.0f}")
            print(f"    P(>0): {mc['prob_positive_net']:.1%}")

    print()
    print("=" * 60)
    print("ROBUSTNESS")
    print("=" * 60)
    print(f"  RT beats Do Nothing in {robustness['vs_baseline']:.0%} of scenarios ({robustness['n_beats_baseline']}/{robustness['n_scenarios']})")
    print(f"  RT beats Max Probability in {robustness['vs_max_prob']:.0%} of scenarios ({robustness['n_beats_max_prob']}/{robustness['n_scenarios']})")

    print()
    print("=" * 60)
    print("WORST CASE")
    print("=" * 60)
    print(f"  Scenario: {worst_case['scenario']}")
    print(f"  Description: {worst_case['description']}")
    print(f"  Net Revenue: Rs.{worst_case['net_revenue']:,.0f}")
    print(f"  Incremental over Do Nothing: Rs.{worst_case['incremental_over_do_nothing']:,.0f}")
    print(f"  Policy Regret: {worst_case['policy_regret']:.1%}")
    print(f"  Beats Do Nothing: {worst_case['beats_do_nothing']}")

    print()
    print("=" * 60)
    print("BREAK-EVEN ANALYSIS")
    print("=" * 60)
    for be in breakeven:
        label = ACTION_LABELS[be["action"]]
        print(f"  {label:<25} break-even at Rs.{be['breakeven_cost']:.2f}")

    print()
    print("=" * 60)
    print("DEGRADATION THRESHOLDS")
    print("=" * 60)
    # Find degradation level where RT stops beating do-nothing
    for d in deg_results:
        if not d.get("rt_incremental", 0) > 0:
            print(f"  RT stops beating Do Nothing at degradation >= {d['degradation']:.0%}")
            break
    else:
        print(f"  RT beats Do Nothing across all tested degradation levels (0-50%)")

    for d in deg_results:
        if not d.get("recoverytwin", 0) > d.get("max_probability", 0):
            print(f"  RT stops beating Max Probability at degradation >= {d['degradation']:.0%}")
            break
    else:
        print(f"  RT beats Max Probability across all tested degradation levels (0-50%)")

    print()
    print("=" * 60)
    print("SEGMENT ANALYSIS (by failure reason)")
    print("=" * 60)
    if seg_failure:
        print(f"  {'Failure Reason':<25} {'N':>5} {'RT Revenue':>12} {'Oracle':>12} {'Regret':>8}")
        print("  " + "-" * 65)
        for s in sorted(seg_failure, key=lambda x: -x.get("regret", 0)):
            print(f"  {str(s['segment_value'])[:25]:<25} {s['n']:>5} "
                  f"Rs.{s['recoverytwin']:>10,.0f} Rs.{s['oracle']:>10,.0f} "
                  f"{s['regret']:>7.1%}")

    print()
    print("=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)
    for check, result in leakage.get("checks", {}).items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {check}")
    if leakage["blocked_features_found"]:
        print(f"  [INFO] Blocked features in data (OK for evaluation): {len(leakage['blocked_features_found'])}")
    print(f"  Overall: {'[PASS]' if leakage['pass'] else '[FAIL]'}")

    # ================================================================
    # VERIFICATION
    # ================================================================
    print()
    print("=" * 60)
    print("PHASE 8 VERIFICATION")
    print("=" * 60)

    rt_val = baseline_policies["recoverytwin"]["net_revenue"]
    dn_val = baseline_policies["do_nothing"]["net_revenue"]

    checks = [
        ("Baseline policies evaluated", len(baseline_policies) >= 4),
        ("Scenarios run", len(scenario_results) >= 5),
        ("Cost sensitivity", len(cost_results) >= 3),
        ("Degradation sensitivity", len(deg_results) >= 3),
        ("Time discount sensitivity", len(td_results) >= 3),
        ("Retry limit sensitivity", len(rl_results) >= 3),
        ("Monte Carlo completed", len(mc_results) >= 3),
        ("Break-even calculated", len(breakeven) >= 2),
        ("Robustness score computed", robustness.get("n_scenarios", 0) >= 5),
        ("Segment analysis", len(segment_results) >= 2),
        ("No leakage", leakage["pass"]),
        ("RT > Do Nothing", rt_val > dn_val),
        ("Reports generated", (Path("reports/phase8/phase8_report.json")).exists()),
    ]

    all_pass = True
    for name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("=" * 60)
        print("PHASE 8: GO")
        print("=" * 60)
    else:
        print("=" * 60)
        print("PHASE 8: NO-GO")
        print("=" * 60)

    print()
    print(f"Reports saved to reports/phase8/")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
