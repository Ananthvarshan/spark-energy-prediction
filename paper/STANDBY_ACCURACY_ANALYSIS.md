# STANDBY Prediction Accuracy — Diagnostic Report

**Question asked:** STANDBY sits at ~74–75% while OFF and the running states are above 90%. Is that acceptable? Does it need to be higher? Can it be improved?

**Scope:** Diagnostic only. No code was changed. Every number below is recomputed from the committed prediction arrays in `outputs/phase5/task12/pelletizer-I/forecasting/preds_*.npz` (5,449 held-out windows × 30 horizon steps = 163,470 step-level predictions), cross-checked against `outputs/phase2/task5/pelletizer-I/comparison_table.csv` and `outputs/phase5/task11/pelletizer-I/forecasting_tests.json`.

---

## 1. First: exactly which number is the 74–75%

Two numbers in the current results match that description, and they mean different things. Both are stated here so the rest of the report is unambiguous.

| Candidate | Value | What it is |
|---|---|---|
| **STANDBY precision (XGBoost, 15 seeds)** | **0.7377 ± 0.0042** | Of the steps the model *calls* STANDBY, 73.8% really are |
| **Macro F1 (XGBoost)** | **0.7530** | Unweighted mean F1 across all four states |

The STANDBY **F1** is **0.689 ± 0.003**, and STANDBY **recall** is **0.647 ± 0.003**. So the honest one-line statement of the current position is:

> The STANDBY class runs at precision 0.74 / recall 0.65 / F1 0.69, against 0.99 for OFF and 0.96 for PEAK_LOAD.

### The full per-class picture (this is the table that should be in the paper, and currently is not)

Pelletizer I, Phase V protocol, XGBoost (the accepted forecaster), mean over 15 seeds:

| State | Mean power | Support (steps) | Class share | Precision | Recall | **F1** | Persistence F1 |
|---|---|---|---|---|---|---|---|
| OFF | 0.4 W | 49,120 | 30.0% | 0.988 | 0.993 | **0.990** | 0.990 |
| PEAK_LOAD | 83,097 W | 97,012 | 59.3% | 0.948 | 0.975 | **0.961** | 0.952 |
| **STANDBY** | **4,209 W** | **10,965** | **6.7%** | **0.738** | **0.647** | **0.689** | 0.681 |
| WORKING (ramp band) | 38,400 W | 6,373 | 3.9% | 0.445 | 0.329 | **0.378** | 0.314 |

Overall accuracy 0.933, macro F1 0.755.

> [!IMPORTANT]
> **There is a worse problem than STANDBY that the current reporting hides.** The `WORKING` state — the 38 kW transitional band between standby and full production — sits at **F1 = 0.378**. It is roughly half as good as STANDBY, it is 3.9% of the record, and it does not appear anywhere in the paper's headline metrics because the paper reports only accuracy, macro F1, and STANDBY F1. If a reviewer recomputes the per-class table, this is the first thing they will find. It should be disclosed before someone else finds it.

---

## 2. The comparison "STANDBY 74% vs OFF/ON 90%+" is not a like-for-like comparison

Three separate things make the STANDBY number structurally lower, and none of them are model defects. They need to be separated before asking whether 74% is "good".

### 2a. The states are not equally frequent

| Class | Share of horizon steps |
|---|---|
| PEAK_LOAD | 59.3% |
| OFF | 30.0% |
| STANDBY | 6.7% |
| WORKING | 3.9% |

OFF and PEAK_LOAD together are **89.3%** of all steps. A classifier that never predicted STANDBY or WORKING at all would already score 89.3% accuracy. The two classes that score above 0.96 are the two that carry almost all the probability mass; the two that score below 0.70 are the two rare ones. Per-class F1 on a 6.7%-prior class is not comparable to per-class F1 on a 59%-prior class — that is a property of the metric, not of the model.

### 2b. STANDBY *detection* is at 93%. The 69% is a **five-minute-ahead forecast**, not a detection score

This is the single most important fact in this report. The forecaster predicts 30 steps of 10 s each = a 300 s horizon. F1 is pooled across all 30 steps. Decomposed by lead time:

| Lead time | XGBoost STANDBY F1 | Persistence STANDBY F1 |
|---|---|---|
| **+10 s (nowcast)** | **0.934** | 0.945 |
| +50 s | 0.836 | 0.836 |
| +60 s | 0.809 | 0.811 |
| +100 s | 0.758 | 0.758 |
| +150 s | 0.676 | 0.667 |
| +200 s | 0.603 | 0.612 |
| +250 s | 0.557 | 0.557 |
| **+300 s** | **0.477** | 0.489 |
| **Pooled (reported)** | **0.678** | 0.682 |

Recomputed directly as a binary STANDBY-vs-rest problem at the nowcast step: **precision 0.910, recall 0.958, F1 0.934.**

> **Identifying STANDBY right now is a 93% problem. Predicting whether the machine will still be in STANDBY in five minutes is a 48% problem. The reported 0.69 is the average of the two, and 0.74 precision is the average of a good near-term number and a bad far-term one.**

That is not a flaw. It is the correct behaviour of any forecaster on a horizon longer than the process's predictability. But it means the sentence "STANDBY accuracy is 74%" is misleading as written, in both directions — it understates detection and overstates 5-minute forecasting.

For reference, the state layer itself (the GMM-HMM labeller, upstream of the forecaster) agrees with the independent Otsu-threshold method (M1) at **99.4%** on this machine and passes **5/5** physics checks (`outputs/phase4/task8/pelletizer-I/characterisation.json`). The state identification of STANDBY is not the weak link.

### 2c. STANDBY sits physically between two neighbours, OFF and PEAK_LOAD do not

STANDBY is 4.2 kW, between OFF (0.4 W) and WORKING (38 kW). OFF is separated from everything by four orders of magnitude in power; PEAK_LOAD is separated by its own 83 kW plateau. STANDBY is the only state with a near neighbour on both sides, and one of those neighbours (`WORKING`, the ramp band) is itself the worst-classified state in the set. **Some of STANDBY's error is inherited from the fact that the state next to it is poorly resolved.**

---

## 3. Where the STANDBY errors actually are

### 3a. Misses (3,933 steps missed out of 10,965 true STANDBY)

| Called instead | Share of misses |
|---|---|
| PEAK_LOAD | 65.5% |
| WORKING | 22.0% |
| OFF | 12.4% |

### 3b. False alarms (2,528 steps wrongly called STANDBY)

| Actually was | Share of false alarms |
|---|---|
| **WORKING (ramp band)** | **63.3%** |
| PEAK_LOAD | 26.0% |
| OFF | 10.7% |

**Nearly two-thirds of false STANDBY calls are the ramp band.** The model is confusing "machine is spooling up/down through 38 kW" with "machine is idling at 4 kW". Those two are physically adjacent in time — the ramp band is what a machine passes through on its way into and out of standby — so the confusion is a **timing** error dressed up as a **class** error.

### 3c. Confirming that: 60.6% of STANDBY errors are timing, not classification

Every horizon cell was tagged with its distance to the nearest true state transition:

| Error location | Share of STANDBY-involved errors (XGBoost) |
|---|---|
| Within 20 s of a real transition | **39.4%** |
| 30–50 s from a real transition | **21.2%** |
| More than 50 s from any transition (genuine bulk error) | 39.4% |

Only **39.4%** of the error is the model being wrong about the machine's state in a stretch where nothing is happening. The other **60.6%** is the model getting the *moment* of a real transition wrong by a few tens of seconds.

Scoring with an explicit timing tolerance makes this concrete:

| Timing tolerance allowed | STANDBY F1 | Precision | Recall |
|---|---|---|---|
| 0 s (as currently reported) | 0.685 | 0.736 | 0.641 |
| ±10 s | 0.703 | 0.763 | 0.652 |
| ±30 s | 0.727 | 0.799 | 0.666 |
| ±50 s | 0.745 | 0.826 | 0.679 |
| ±100 s | 0.779 | 0.878 | 0.700 |

> Allowing the model to be **50 seconds early or late** on a transition — well inside the 600 s minimum-off time the decision layer already enforces — lifts STANDBY precision from 0.736 to **0.826**. The decision layer does not care about 50 s. The metric does.

---

## 4. Is 74% fine? — the ceiling analysis

The question "should it be higher" only has an answer relative to what is attainable. Two measurements bound it.

### 4a. The model is at 99.3% of the perfect-current-state ceiling

Construct an oracle that **knows the true state of the machine right now with zero error** and simply holds it across the 300 s horizon. That oracle is not achievable in deployment; it is an upper bound on everything a model can extract from present-state information alone.

| Predictor | STANDBY F1 |
|---|---|
| **Oracle: perfect current state, held for 300 s** | **0.6945** (P 0.701 / R 0.688) |
| XGBoost (actual) | **0.6893** |
| Persistence (last *observed* state, held) | 0.6815 |

**XGBoost is at 99.3% of an oracle that cannot be built.** Every remaining point of STANDBY F1 must come from *anticipating transitions the machine has not made yet* — not from better state recognition, better features on the current sample, or a bigger model. There is essentially nothing left in the "recognise the present better" direction.

### 4b. The horizon is mostly unpredictable by construction

| Property of the test set | Value |
|---|---|
| Windows where the true state never changes across 300 s | **85.6%** |
| Mean true state changes per 300 s window | **0.414** |
| Windows containing any STANDBY at all | 13.5% |
| Windows that are entirely STANDBY | 2.9% |

In 85.6% of windows the correct answer is "nothing changes", which persistence gets for free. All achievable skill is concentrated in the 14.4% of windows containing a transition, and STANDBY is present in only 13.5% of windows to begin with. This is why **five of the seven models in the comparison lose to doing no training at all**, and why the winning model beats persistence by 1.45 s of MAE and 0.008 of F1 — findings the paper already documents in `paper/forecasting.tex`.

### 4c. A realistic ceiling estimate

Combining §4a and §3c: the achievable range for pooled STANDBY F1 on this record, with a perfect transition-timing model, is approximately **0.78–0.85** — the ±100 s tolerance figure (0.779) is roughly what a model that gets transitions right to within a minute-and-a-half would score at zero tolerance. **0.95 is not on the table** for a 300 s-ahead forecast on a process with 0.41 transitions per window.

**Verdict on "is it fine":** Yes, 0.74 precision / 0.69 F1 is a defensible number for this task, **provided it is reported as what it is** — a 300 s-ahead forecast of a 6.7%-prior transitional state, at 99.3% of the present-state ceiling. It is not defensible if reported as "STANDBY detection accuracy", because that number is 93%.

---

## 5. Does improving it actually buy anything?

This is the question that determines whether it is worth spending effort. The evidence in the repo says **no**, and says it clearly.

### 5a. The F1 → dollars transfer function is nearly flat

From `outputs/phase6/task13e/pelletizer-I/task13e_results.json` and `outputs/phase5/task11/pelletizer-I/decision_bootstrap_S2_moderate.json` (S2, moderate scenario, 394 held-out episodes over 73.6 days):

| Change | STANDBY F1 delta | Annual saving delta |
|---|---|---|
| Persistence → XGBoost (best available forecaster) | +0.008 | +1.45 s MAE, p_Holm = 7×10⁻⁴ |
| Adding the chance constraint | — | **+17.8 USD/yr** |
| Targeting the conditional mean instead of median | — | **+31.5 USD/yr** |
| Adaptive optimisation instead of the static break-even rule | — | **+68.9 USD/yr** |

And the proposed policy's own bootstrap interval:

| Policy | Saving | 95% CI |
|---|---|---|
| `forecast_opt` (proposed) | +2.86 USD/yr | **[−55.9, +46.2]** |
| `ski_rental` (no forecast at all) | +0.02 USD/yr | [−25.0, +28.8] |

> The confidence interval on the decision layer is **±50 USD/yr around a 2.86 USD/yr point estimate**. Three *structural* choices (chance constraint, estimand, adaptive-vs-static) are each worth 18–69 USD/yr. Nothing achievable by raising STANDBY F1 from 0.69 to 0.75 registers against that noise floor.

### 5b. The window-level error the decision layer actually consumes

| Metric | XGBoost | Persistence |
|---|---|---|
| STANDBY-seconds MAE, all windows | 11.63 s | 12.95 s |
| …on the 13.5% of windows containing STANDBY | 79.65 s | 88.38 s |
| …on the 86.5% with no STANDBY (false-alarm cost) | **1.04 s** | 1.21 s |

The false-alarm cost on non-STANDBY windows is **1 second out of 300**. The decision layer's minimum-off time is 600 s. The forecaster's errors are already an order of magnitude below the granularity at which the policy acts.

**Conclusion: STANDBY F1 is not the binding constraint on this system. It has not been the binding constraint since Phase III.**

---

## 6. Can it be improved anyway? — ranked options

If the number must go up (for reviewer optics or for a different deployment), these are the levers, ordered by expected return per unit of effort. Effect sizes are estimates grounded in the measurements above.

| # | Lever | Expected STANDBY gain | Cost | Assessment |
|---|---|---|---|---|
| **1** | **Report per-horizon, not pooled** | 0.69 → **0.93** at the operationally relevant lead time | Zero — reporting change only | **Do this.** It is not an improvement, it is a correction to a misleading aggregate. |
| **2** | **Add elapsed-dwell as a feature** | est. +0.02 to +0.05 | One retrain | **Genuine gap — see below.** |
| **3** | **Move the decision threshold** | P 0.738 / R 0.647 → balanced ≈ 0.70 F1 | Trivial | Small but free; the model is currently under-calling STANDBY. |
| **4** | **Replace per-step classification with a duration/hazard model** | est. +0.05 to +0.10 | Substantial rework | Best *technical* option; see below. |
| **5** | Report with a stated timing tolerance | 0.685 → 0.745 at ±50 s | Zero | Legitimate if pre-registered and justified by the 600 s min-off constraint. Looks like metric-shopping if added after the fact. |
| **6** | Merge STANDBY + WORKING into one "energised non-productive" class | 0.689 → **0.733** | Zero | **Do not.** It destroys the physics semantics the whole framework rests on. Listed for completeness. |
| **7** | Bigger / different neural architecture | **Negative** | High | Already tested. All five neural models lose to persistence; Seq2Seq LSTM is worst of seven. |
| **8** | More training data from other machines | **Negative** | High | Already tested in Phase IV. Cross-type transfer loses 86% of STANDBY F1; cross-dataset Spearman is −0.245. |

### Detail on #2 — elapsed dwell is missing, and this is a real finding

The Phase V forecaster's feature list (`outputs/phase5/task12/pelletizer-I/forecasting/forecasting_seeds.json`) contains 24 columns: 9 electrical, 8 rolling statistics, and 7 calendar. It does **not** contain elapsed dwell.

The earlier 3-class model in `outputs/models/pelletizer-I/training_config.json` **did** carry `dwell_seconds_so_far` as one of its 13 features. That feature was dropped somewhere between the original pipeline and the Phase II/V baseline harness.

This matters because "how long has the machine already been in this state" is the single strongest predictor of "how much longer will it stay" for any process with non-exponential dwell times — and this record has exactly that: median dwell 92.5 s with a heavy tail out to hours. The decision-layer forecaster uses elapsed idle time and gets Spearman 0.672 from it; the window forecaster does not have access to the equivalent signal. This is the one concrete, cheap, defensible improvement available.

*(Stated as an observation, not an action — no code was changed.)*

### Detail on #4 — the task is framed as the wrong problem

Per-step multiclass classification over a fixed 300 s horizon asks the model 30 independent questions when the underlying quantity is one number: **time until the next state change**. The consequences are visible in the diagnostics — 60.6% of errors are transition-timing errors, and the errors are strongly correlated across horizon steps within a window (once the model gets a transition time wrong, every subsequent step is wrong too, which is why F1 decays monotonically from 0.934 to 0.477).

A hidden semi-Markov / survival formulation predicts the dwell distribution directly and would score naturally on the quantity the decision layer consumes. The paper already cites `yu2010hsmm` in `related_work.tex` §2.1 for exactly this reason. This is the principled fix, and it is a Phase VIII/IX-sized piece of work, not a tweak.

---

## 7. Recommendations

**On whether the number is acceptable — yes, with three reporting corrections:**

1. **Split detection from forecasting.** State the nowcast number (STANDBY F1 = 0.934, precision 0.910, recall 0.958) separately from the 300 s-ahead number (0.689). Reporting only the pooled figure understates the state layer and overstates the forecaster. This is the single highest-value change and costs nothing.

2. **Publish the full per-class table.** Add OFF / PEAK_LOAD / STANDBY / WORKING precision-recall-F1 with support columns. Two reasons: it makes the class-prior explanation self-evident (6.7% vs 59.3%), and it discloses the WORKING = 0.378 result on your terms rather than a reviewer's.

3. **Quote the ceiling.** "XGBoost reaches 99.3% of the STANDBY F1 attainable by an oracle with perfect knowledge of the current state" is a much stronger sentence than "STANDBY F1 is 0.689", and it is measured, not asserted. It also pre-empts the obvious reviewer question — *why didn't you try harder on the minority class?* — with a number.

**On whether to spend effort raising it — no, with one exception:**

The economics say the forecaster is not the bottleneck (§5). The three decision-layer structural choices are each worth 6–24× more than the entire persistence-to-XGBoost improvement, and the bootstrap interval swamps everything. Adding elapsed dwell (#2) is worth doing because it is cheap and closes a genuine gap, not because the dollars require it.

**The claim the results will support:**

> STANDBY is identified at F1 = 0.93 and forecast 300 s ahead at F1 = 0.69, which is 99.3% of the ceiling set by perfect knowledge of the present state. Sixty percent of the residual error is transition-timing error within 50 s, inside the 600 s minimum-off time the policy enforces, and therefore does not propagate to the decision.

That is defensible, measured end-to-end, and consistent with everything already in Phases II–V.

---

## Appendix — sources

| Claim | File |
|---|---|
| Per-class metrics, confusions, horizon decomposition, tolerance curves, oracle ceiling | recomputed from `outputs/phase5/task12/pelletizer-I/forecasting/preds_*.npz` |
| Headline model comparison | `outputs/phase2/task5/pelletizer-I/comparison_table.csv` |
| Seed spreads, significance | `outputs/phase5/task11/pelletizer-I/forecasting_tests.json` |
| Component value in USD/yr | `outputs/phase6/task13e/pelletizer-I/task13e_results.json` |
| Decision-layer bootstrap CIs | `outputs/phase5/task11/pelletizer-I/decision_bootstrap_S2_moderate.json` |
| State powers, shares, physics score, M1 agreement | `outputs/phase4/task8/pelletizer-I/characterisation.json`, `outputs/phase4/task8/table_state_identification.csv` |
| Feature lists (dwell-feature gap) | `outputs/phase5/.../forecasting_seeds.json` vs `outputs/models/pelletizer-I/training_config.json` |
