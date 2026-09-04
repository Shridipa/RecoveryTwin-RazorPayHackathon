# RecoveryTwin — 5-Minute Pitch Script

## Razorpay AI Revenue Recovery Hackathon

---

### SLIDE 1: THE PROBLEM (0:00 – 0:45)

**[Show: Landing page hero — "Recover more from every failed payment"]**

> Every day, millions of payments fail on Indian payment platforms.
> Some fail because of a temporary bank timeout.
> Some fail because the customer's card expired.
> Some fail because of a network glitch.
>
> The question is not *whether* to recover — it's *how*.
>
> Should you retry the same method? Send a reminder? Offer an alternative payment method? Or do nothing?
>
> Razorpay processes billions in payments. Even a 1% improvement in recovery rate translates to crores in recovered revenue.
>
> But today, most recovery systems use the same action for every failed payment. They don't consider the customer, the failure reason, the payment amount, or the cost of intervention.
>
> **RecoveryTwin changes that.**

---

### SLIDE 2: THE SOLUTION (0:45 – 1:45)

**[Show: Click "Open Recovery Center" → Command Center dashboard]**

> RecoveryTwin is a counterfactual ML system that answers one question for every failed payment:
>
> **"What is the single best action Razorpay should take right now to maximize expected recovered revenue?"**
>
> It does this in five steps:
>
> **First** — it predicts the probability of recovery for each possible action. Not just "will this payment recover?" but "what happens if we retry vs. remind vs. switch methods?"
>
> **Second** — it calibrates those probabilities so they're financially trustworthy. A 70% probability actually means 70 out of 100 similar payments recover.
>
> **Third** — it estimates *when* recovery will happen, because a payment recovered in 2 hours is worth more than one recovered in 72 hours.
>
> **Fourth** — it calculates the *expected financial value* of each action: probability × payment amount − intervention cost.
>
> **Fifth** — it applies real-world constraints: customer fatigue limits, retry caps, cost thresholds, and compliance rules.

**[Show: Navigate to a Payment Detail page]**

> For this ₹8,500 failed payment, RecoveryTwin evaluates four options side by side and recommends Retry — because it has the highest expected value after accounting for cost and recovery timing.

---

### SLIDE 3: THE ENGINE (1:45 – 2:45)

**[Show: Model Performance page]**

> Under the hood, RecoveryTwin is not one model — it's a pipeline of six specialized ML components:
>
> 1. **Recovery Predictor** — XGBoost and LightGBM models trained on 50,000 synthetic transactions with treatment-aware features. We achieved a PR-AUC of 0.516 and ROC-AUC of 0.709.
>
> 2. **Probability Calibrator** — Sigmoid and isotonic calibration with an ECE of 0.012. When the model says 70%, it means 70%.
>
> 3. **Causal / Uplift Model** — S-Learner, T-Learner, X-Learner, and R-Learner implementations that estimate *individual* treatment effects. Not just averages — per-payment uplift.
>
> 4. **Survival Model** — Cox Proportional Hazards and Random Survival Forests that estimate time-to-recovery. C-index of 0.665.
>
> 5. **Decision Engine** — Combines all models into a counterfactual financial optimizer that respects fatigue, retry limits, and cost constraints.
>
> 6. **Financial Simulator** — Monte Carlo simulation with 1,000 iterations across 12 economic scenarios to stress-test the policy.

**[Show: Scenario Lab page]**

> We tested RecoveryTwin under 12 different economic conditions — high costs, low recovery rates, treatment degradation, adverse combinations.
>
> **RecoveryTwin outperforms "Do Nothing" in 100% of scenarios.**

---

### SLIDE 4: THE RESULTS (2:45 – 3:45)

**[Show: Command Center — metrics]**

> On 8,426 unseen test payments:
>
> - **Do Nothing** recovers ₹18.2 lakhs
> - **Always Retry** recovers ₹32.2 lakhs
> - **Max Probability** — the strongest baseline — recovers ₹42.1 lakhs
> - **RecoveryTwin** recovers ₹38.2 lakhs — that's ₹20 lakhs more than doing nothing
>
> Now, you might ask: why is RecoveryTwin below Max Probability?
>
> We investigated this rigorously. We found a **critical outcome leakage bug** — the decision engine was using the observed recovery status to decide whether to intervene, which is impossible in production. We fixed it, and recovery improved by 36%.
>
> The remaining gap comes from the S-Learner's noisy individual-level CATE estimates. The level predictions are well-calibrated; the differences between them are not.
>
> **We chose to report this honestly rather than hide it.** That's how real ML engineering works.

**[Show: Financial Analysis page — Monte Carlo distribution]**

> Our Monte Carlo simulation shows 100% probability of positive net revenue across all scenarios. The P5 floor is ₹32 lakhs. Even in the worst case, RecoveryTwin creates value.

---

### SLIDE 5: THE DIFFERENTIATOR (3:45 – 4:30)

**[Show: Landing page product visualization]**

> What makes RecoveryTwin different from a simple "retry on failure" system?
>
> **Three things:**
>
> **First — it's counterfactual.** It doesn't just predict recovery. It asks "what would happen under each possible action?" That's causal inference, not just prediction.
>
> **Second — it's financially grounded.** Every decision is backed by expected monetary value. Not accuracy. Not AUC. Real rupees.
>
> **Third — it's stress-tested.** We didn't just train a model and deploy it. We simulated what happens when recovery rates drop 30%, when intervention costs rise 50%, when customer fatigue increases.
>
> RecoveryTwin is not an ML demo. It's a **financial decision system**.

---

### SLIDE 6: THE DEMO (4:30 – 5:00)

**[Show: Live dashboard walkthrough]**

> Let me show you the product.
>
> This is the Command Center — ₹11.5 lakhs at risk, ₹3.8 lakhs expected recovery.
>
> This is the Payment Queue — 8,426 failed payments, each with a recommended action.
>
> Let me click on this ₹42,500 payment. Here's the counterfactual analysis — four actions compared side by side. Retry has the highest expected value. The system explains *why* — failure pattern is recoverable, customer has strong history, fatigue allows intervention.
>
> Now let me stress-test — what if retry effectiveness drops 25%? The Scenario Lab shows RecoveryTwin remains profitable.
>
> 154 tests passing. 17 verification sections. Zero data leakage. Full audit trail.
>
> **RecoveryTwin — predict the loss, simulate the alternatives, recover the money.**

---

## TIMING GUIDE

| Section | Duration | What to show |
|---------|----------|-------------|
| Problem | 45 sec | Landing page hero |
| Solution | 60 sec | Dashboard → Payment Detail |
| Engine | 60 sec | Model Performance → Scenario Lab |
| Results | 60 sec | Command Center → Financial Analysis |
| Differentiator | 45 sec | Landing page product visualization |
| Demo | 30 sec | Live walkthrough |
| **Total** | **5:00** | |

## KEY TALKING POINTS

- "Counterfactual ML" — not just prediction, but "what if we do X?"
- "Financially grounded" — expected value, not accuracy
- "Stress-tested" — 12 scenarios, Monte Carlo, worst-case analysis
- "Leakage audit" — found and fixed a real production bug
- "154 tests, 17 verification sections" — engineering rigor
- "₹20 lakhs incremental recovery" — concrete business value

## COMMON JUDGE QUESTIONS

**Q: Why is RecoveryTwin below Max Probability?**
> The S-Learner's individual-level CATE estimates have 22% sign agreement. Level predictions are well-calibrated; differences are noisy. We chose honest reporting over inflated metrics.

**Q: Is this real data?**
> No — it's a synthetic payment environment calibrated using public UPI statistics and payment failure datasets. No Razorpay production data was used.

**Q: How does this handle cold-start / new merchants?**
> The model uses merchant-level features (transaction count, recovery rate, average amount). New merchants would fall back to population-level priors.

**Q: What about adversarial customers who game the system?**
> Fatigue limits and retry caps prevent over-intervention. The policy constraints are configurable per merchant segment.

**Q: How would this work in production?**
> The FastAPI backend exposes all decision logic through REST APIs. The decision engine processes payments in real-time. The financial simulator runs offline for policy evaluation.
