"""
RecoveryTwin Financial Policy Simulator — Report Generation.

Generates machine-readable JSON and human-readable markdown reports.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


def save_json_report(
    data: Dict[str, Any],
    path: str,
) -> None:
    """Save a dictionary as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def generate_phase8_report(
    baseline_policies: Dict[str, Dict],
    scenario_results: List[Dict],
    cost_sensitivity: List[Dict],
    degradation_sensitivity: List[Dict],
    time_discount_sensitivity: List[Dict],
    retry_limit_sensitivity: List[Dict],
    monte_carlo_results: Dict[str, Any],
    breakeven_results: List[Dict],
    robustness: Dict[str, float],
    worst_case: Dict[str, Any],
    segment_results: List[Dict],
    leakage_audit: Dict[str, Any],
    report_dir: str = "reports/phase8",
) -> Dict[str, Any]:
    """Generate the full Phase 8 report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "phase": 8,
        "title": "Financial Policy Simulation & Stress Testing",
        "baseline_policies": baseline_policies,
        "scenario_results": scenario_results,
        "cost_sensitivity": cost_sensitivity,
        "degradation_sensitivity": degradation_sensitivity,
        "time_discount_sensitivity": time_discount_sensitivity,
        "retry_limit_sensitivity": retry_limit_sensitivity,
        "monte_carlo": monte_carlo_results,
        "breakeven_analysis": breakeven_results,
        "robustness": robustness,
        "worst_case": worst_case,
        "segment_analysis": segment_results,
        "leakage_audit": leakage_audit,
    }

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full JSON report
    save_json_report(report, str(out_dir / "phase8_report.json"))

    # Save individual components
    save_json_report(baseline_policies, str(out_dir / "baseline_policies.json"))
    save_json_report({"scenarios": scenario_results}, str(out_dir / "scenario_results.json"))
    save_json_report({"cost_sensitivity": cost_sensitivity}, str(out_dir / "cost_sensitivity.json"))
    save_json_report({"degradation_sensitivity": degradation_sensitivity}, str(out_dir / "degradation_sensitivity.json"))
    save_json_report(monte_carlo_results, str(out_dir / "monte_carlo_results.json"))
    save_json_report({"breakeven": breakeven_results}, str(out_dir / "breakeven_results.json"))
    save_json_report(robustness, str(out_dir / "robustness.json"))
    save_json_report(worst_case, str(out_dir / "worst_case.json"))
    save_json_report(leakage_audit, str(out_dir / "leakage_audit.json"))

    # Save segment analysis as CSV
    if segment_results:
        import pandas as pd
        pd.DataFrame(segment_results).to_csv(
            str(out_dir / "segment_analysis.csv"), index=False
        )

    # Generate markdown report
    _generate_markdown_report(report, out_dir)

    return report


def _generate_markdown_report(report: Dict[str, Any], out_dir: Path) -> None:
    """Generate a human-readable markdown report."""
    lines = [
        "# RecoveryTwin — Phase 8: Financial Policy Simulation",
        "",
        f"*Generated: {report['timestamp']}*",
        "",
        "## Baseline Policy Comparison",
        "",
        "| Policy | Net Revenue | Recovery Rate | Interventions |",
        "|--------|------------|---------------|---------------|",
    ]

    for name, pol in report.get("baseline_policies", {}).items():
        lines.append(
            f"| {name} | Rs.{pol.get('net_revenue', 0):,.0f} | "
            f"{pol.get('recovery_rate', 0):.1%} | {pol.get('n_interventions', 0)} |"
        )

    lines.extend([
        "",
        "## Scenario Results",
        "",
        "| Scenario | RT Net Revenue | RT Incremental | Policy Regret | Beats Baseline |",
        "|----------|---------------|----------------|---------------|----------------|",
    ])

    for s in report.get("scenario_results", []):
        rt_rev = s["policies"].get("recoverytwin", {}).get("net_revenue", 0)
        lines.append(
            f"| {s['scenario']} | Rs.{rt_rev:,.0f} | "
            f"Rs.{s.get('recoverytwin_incremental', 0):,.0f} | "
            f"{s.get('policy_regret', 0):.1%} | "
            f"{'Yes' if s.get('beats_do_nothing') else 'No'} |"
        )

    lines.extend([
        "",
        "## Cost Sensitivity (Retry Cost → RT Net Revenue)",
        "",
        "| Retry Cost | RT Net Revenue | Incremental vs Do Nothing |",
        "|------------|---------------|---------------------------|",
    ])

    for s in report.get("cost_sensitivity", []):
        lines.append(
            f"| Rs.{s['retry_cost']:.2f} | Rs.{s['recoverytwin']:,.0f} | "
            f"Rs.{s.get('rt_incremental', 0):,.0f} |"
        )

    lines.extend([
        "",
        "## Treatment Degradation",
        "",
        "| Degradation | RT Net Revenue | Policy Regret |",
        "|-------------|---------------|---------------|",
    ])

    for d in report.get("degradation_sensitivity", []):
        lines.append(
            f"| {d['degradation']:.0%} | Rs.{d['recoverytwin']:,.0f} | "
            f"{d.get('policy_regret', 0):.1%} |"
        )

    lines.extend([
        "",
        "## Monte Carlo Summary",
        "",
    ])

    mc = report.get("monte_carlo", {})
    for pol_name, pol_mc in mc.items():
        if isinstance(pol_mc, dict) and "mean_net_revenue" in pol_mc:
            lines.append(f"### {pol_name}")
            lines.append(f"- Mean: Rs.{pol_mc['mean_net_revenue']:,.0f}")
            lines.append(f"- P5: Rs.{pol_mc.get('p5', 0):,.0f}")
            lines.append(f"- P95: Rs.{pol_mc.get('p95', 0):,.0f}")
            lines.append(f"- P(positive): {pol_mc.get('prob_positive_net', 0):.1%}")
            lines.append("")

    lines.extend([
        "## Break-Even Analysis",
        "",
    ])

    for be in report.get("breakeven_analysis", []):
        lines.append(f"- Action {be['action']}: break-even at Rs.{be['breakeven_cost']:.2f}")

    robustness = report.get('robustness', {})
    worst_case = report.get('worst_case', {})
    lines.extend([
        "",
        "## Robustness",
        "",
        f"- RecoveryTwin beats Do Nothing in **{robustness.get('vs_baseline', 0):.0%}** of scenarios",
        f"- RecoveryTwin beats Max Probability in **{robustness.get('vs_max_prob', 0):.0%}** of scenarios",
        "",
        "## Worst Case",
        "",
        f"- Scenario: {worst_case.get('scenario', 'N/A')}",
        f"- Net Revenue: Rs.{worst_case.get('net_revenue', 0):,.0f}",
        f"- Beats Do Nothing: {worst_case.get('beats_do_nothing', False)}",
        "",
        "## Leakage Audit",
        "",
    ])

    for check, result in report.get("leakage_audit", {}).items():
        status = "[OK]" if result else "[FAIL]"
        lines.append(f"- {status} {check}")

    with open(out_dir / "phase8_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
