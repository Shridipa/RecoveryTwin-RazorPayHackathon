# RecoveryTwin — Phase 8: Financial Policy Simulation

*Generated: 2026-08-24T22:12:04.438072*

## Baseline Policy Comparison

| Policy | Net Revenue | Recovery Rate | Interventions |
|--------|------------|---------------|---------------|
| do_nothing | Rs.1,822,926 | 17.4% | 0 |
| always_retry | Rs.3,217,224 | 29.2% | 8426 |
| max_probability | Rs.4,214,668 | 39.0% | 8414 |
| recoverytwin | Rs.3,822,601 | 37.1% | 7834 |
| oracle | Rs.7,315,627 | 65.9% | 4084 |

## Scenario Results

| Scenario | RT Net Revenue | RT Incremental | Policy Regret | Beats Baseline |
|----------|---------------|----------------|---------------|----------------|
| BASELINE | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| LOW_RECOVERY | Rs.3,822,601 | Rs.2,546,553 | 46.6% | Yes |
| HIGH_RECOVERY | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| HIGH_COST | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| LOW_COST | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| HIGH_FATIGUE | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| LOW_FATIGUE | Rs.3,822,601 | Rs.1,999,675 | 47.8% | Yes |
| HIGH_VALUE | Rs.3,822,601 | Rs.176,750 | 73.9% | Yes |
| LOW_VALUE | Rs.3,822,601 | Rs.2,911,138 | -4.4% | Yes |
| DEGRADATION_20 | Rs.3,822,601 | Rs.1,999,675 | 28.4% | Yes |
| DEGRADATION_40 | Rs.3,822,601 | Rs.1,999,675 | -0.5% | Yes |
| ADVERSE_COMBINED | Rs.3,822,601 | Rs.2,801,763 | -14.9% | Yes |

## Cost Sensitivity (Retry Cost → RT Net Revenue)

| Retry Cost | RT Net Revenue | Incremental vs Do Nothing |
|------------|---------------|---------------------------|
| Rs.0.05 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.0.10 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.0.25 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.0.50 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.1.00 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.2.00 | Rs.3,822,601 | Rs.1,999,675 |
| Rs.5.00 | Rs.3,822,601 | Rs.1,999,675 |

## Treatment Degradation

| Degradation | RT Net Revenue | Policy Regret |
|-------------|---------------|---------------|
| 0% | Rs.3,822,601 | 47.8% |
| 10% | Rs.3,822,601 | 39.1% |
| 20% | Rs.3,822,601 | 28.4% |
| 30% | Rs.3,822,601 | 15.4% |
| 40% | Rs.3,822,601 | -0.5% |
| 50% | Rs.3,822,601 | -19.6% |

## Monte Carlo Summary

### do_nothing
- Mean: Rs.1,822,778
- P5: Rs.1,814,446
- P95: Rs.1,830,550
- P(positive): 100.0%

### always_retry
- Mean: Rs.3,217,427
- P5: Rs.3,207,508
- P95: Rs.3,227,439
- P(positive): 100.0%

### max_probability
- Mean: Rs.4,214,928
- P5: Rs.4,202,807
- P95: Rs.4,226,471
- P(positive): 100.0%

### recoverytwin
- Mean: Rs.4,022,156
- P5: Rs.4,011,346
- P95: Rs.4,034,300
- P(positive): 100.0%

## Break-Even Analysis

- Action 1: break-even at Rs.100.00
- Action 2: break-even at Rs.100.00
- Action 3: break-even at Rs.100.00

## Robustness

- RecoveryTwin beats Do Nothing in **100%** of scenarios
- RecoveryTwin beats Max Probability in **33%** of scenarios

## Worst Case

- Scenario: BASELINE
- Net Revenue: Rs.3,822,601
- Beats Do Nothing: True

## Leakage Audit

- [OK] blocked_features_found
- [FAIL] allowed_features
- [FAIL] unknown_features
- [OK] pass
- [OK] checks