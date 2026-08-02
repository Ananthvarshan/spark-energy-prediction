# Phase V — Statistical analysis: completion notes

Covers Task 11 (significance testing) and Task 12 (repeated runs), plus the
four consistency fixes that had to land before the tests generated final
numbers. Every number below is from runs on this machine.

Reproduce with `python run_phase5.py` (~6 h; `--skip task12` runs the tests
from the stored runs in ~25 min).

## Files produced

| File | Role |
|---|---|
| `src/stats.py` | bootstrap, paired tests, corrections, effect sizes, power |
| `experiments/task12_multiple_runs.py` | Task 12 — repeated runs (forecasting, labelling, decision) |
| `experiments/task11_significance.py` | Task 11 — all significance testing |
| `experiments/make_figures_phase5.py` | five result figures |
| `run_phase5.py` | master runner |
| `outputs/phase5/**` | per-run metrics, predictions, test tables, figures, logs |

---

## Four fixes applied before the tests ran

These change what the tests are testing, so they had to precede them.

1. **`validate_gmm.py` test T3 renamed.** It was titled "k Selection
   (BIC + physical filters agree)". Phase IV showed they do not agree — BIC is
   still falling at k = 8 on seven of eight machines — so the title now reads
   "k Selection (physics-priority; BIC diagnostic)", in both the module
   docstring and the final report dictionary.
2. **A persistence row was added to Phase II's forecasting table.** The
   comparison ranked six *trained* models and never asked whether training
   helped at all. `make_windows` gained an optional `return_last_state`, the
   model registry gained a `persistence` entry, and Task 5 gained `--append`
   so the row could be added to the existing results without re-running the
   six models — and it refuses to append if the window geometry has changed,
   so the added row is guaranteed to be scored on the same windows.
3. **`paper/decision_layer.tex` now carries both Phase III headline numbers**
   (+38.6 USD/yr at 55% of oracle on the Phase I–III labels; +7.2 USD/yr at
   16.8% under the minimum-dwell labelling), with the episode-count and
   median-duration rows that explain the difference, and the payback paragraph
   quotes both.
4. **`paper/PHASE2_NOTES.md`'s empty Task 5 section is filled in**, including
   the persistence row and what it does to the Phase II claim.

**Fix 2 changed a Phase II conclusion before Phase V even started.** On Phase
II's own protocol and labels, persistence reaches STANDBY F1 **0.6815** and MAE
**12.95 s**, against the proposed Seq2Seq LSTM's **0.6597 / 13.76 s**. The
untrained reference beats every neural model in the table and sits 0.004 F1
below XGBoost. Whether that last gap is real is the first thing Task 11 tests.

**It is.** XGBoost beats persistence by 1.45 s of per-window MAE
(p_holm = 0.0007), which is both statistically detectable and above the 1 s
practical threshold — and it is the *only* comparison in the whole forecasting
family where training beats not training. Task 11A below shows the five neural
models all lose to the untrained reference, the proposed Seq2Seq LSTM by the
largest margin of the seven.

---

## Method: four decisions that determine every interval below

**The resampling unit is never a row.** A 1 Hz record is massively
autocorrelated, and an i.i.d. bootstrap over rows would treat 5.5 million
readings as 5.5 million independent observations. Measured on a synthetic
AR(1) series with ρ = 0.999, the i.i.d. bootstrap reports a standard error 25×
too small. Phase V therefore uses:

| quantity | unit | why |
|---|---|---|
| state-time totals | one-week moving blocks | the duty cycle closes over a week (shift pattern + weekend) |
| decision-layer savings | factory days | the restart cap is per calendar day; episodes within a day share a shift and a production plan |
| forecast errors | test windows, **paired** | the same window attempted by two models is one problem solved twice |

**A day drawn twice counts as two separate days.** In the decision bootstrap
the resampled copies are re-keyed, because leaving the label unchanged would
let two copies share one day's restart budget and silently tighten the
constraint being evaluated.

**Every policy is scored on the same resample.** Drawing independent resamples
per policy would break the pairing and inflate the interval of every contrast —
and the contrasts are what the paper's claims rest on.

**Statistical and economic significance are separate columns.** A paired test
over 5,449 windows detects differences far below anything that changes a
decision, so each comparison carries a p-value, an effect size, and a fixed
practical threshold (1 s of per-window MAE, 0.02 F1, 5 USD/yr). A result is
allowed to be "detectable but immaterial", and several are.

**Bonferroni is not used.** Holm's step-down procedure controls the same
family-wise error rate and is uniformly more powerful; Benjamini–Hochberg is
reported beside it for the exploratory families. Both are shown next to the
raw p-value so the cost of the correction is visible.

---

## Task 11D — cross-machine tests, and the power that is not there

| comparison | n | mean difference (F1) | p | p-value floor | power at the observed effect |
|---|---|---|---|---|---|
| own model vs persistence, all machines | 8 | **+0.194** | **0.0156** | 0.0078 | 0.41 |
| own model vs persistence, motor machines only | 4 | +0.048 | 0.2500 | **0.1250** | **0.00** |
| local vs transferred sequence model | 8 | **+0.384** | **0.0078** | 0.0078 | — (d = +1.75) |

Three things to take from this table.

**Training beats not training across the eight machines — but the effect lives
in the auxiliary plant.** The +0.194 F1 mean difference is dominated by
exhaust-fan-I (0.982 vs 0.288) and dpc-I (0.402 vs 0.012), machines whose
"STANDBY" cluster is really their off state and therefore trivially
predictable once a model has seen it. On the four motor-driven machines — the
ones where the state model is physically valid — the difference is +0.048 F1
and cannot be tested.

**"Cannot be tested" is meant literally.** With four paired observations the
smallest two-sided p-value the signed-rank test can return is 2/2⁴ = **0.125**,
which is above α = 0.05. No data, and no effect size however large, can produce
a significant result at n = 4. A non-significant verdict here is a statement
about the design, not about the models, and the paper must say so in exactly
those terms rather than reporting "no significant difference".

**The transfer degradation is real.** Local models beat transferred ones by
0.384 F1 with p at the floor and d = +1.75. This is the one cross-machine claim
in Phase IV that survives testing outright.

Power curves (Figure `fig_task11_power`) put numbers on what these designs
could have found: at n = 8 the paired test reaches 80% power only for effects
above ~1.4 sd; at n = 4 it never does.

---

## Task 11C — intervals on the state-time totals

One-week moving-block bootstrap of the STANDBY share, per machine:

| machine | STANDBY h/covered day | 95% CI | one-week blocks |
|---|---|---|---|
| pelletizer-I | 1.40 | [1.15, 1.63] | 10 |
| pelletizer-II | 1.85 | [1.47, 2.28] | 9 |
| milling-I | 5.52 | [4.58, 6.23] | **2** |
| milling-II | 5.27 | [4.44, 5.89] | **2** |
| exhaust-fan-I | 8.73 | [7.31, 10.03] | 10 |
| exhaust-fan-II | 1.45 | [1.09, 2.07] | 9 |
| dpc-I | 0.90 | [0.71, 1.12] | 10 |
| dpc-II | 1.16 | [0.85, 1.42] | 9 |

The intervals are ±15–25% of the point estimate on the five-month records:
**the standby totals the paper quotes are stable to about a fifth of their
value, not to the three significant figures they have been written with so
far.** Every state-time number in the paper should be rounded accordingly.

**The two milling machines have two blocks.** Twelve covered days cannot
support a weekly-block interval, and the intervals in their rows are reported
only to make that visible — they are not usable. This is the same limitation
Phase IV recorded from a different direction (1,082 acquisition fragments), and
it is why the milling machines' 5.5 h/day, the largest per-day standby figure
among the motor-driven machines, must not be annualised.

---

## Task 11A — forecasting models

Seven models on the identical 5,449 held-out windows. Every stored prediction
array was re-scored and matched against its recorded metrics before any test
ran, and all seven sit on a byte-identical `y_true`, which is what makes the
pairing legitimate.

| model | seeds | STANDBY F1 (mean ± sd) | min–max | per-window MAE |
|---|---|---|---|---|
| XGBoost | 15 | **0.6893 ± 0.0025** | 0.6847–0.6937 | **11.50 s** |
| Persistence (untrained) | 1 | 0.6815 (deterministic) | — | 12.95 s |
| Transformer | 5 | 0.5945 ± 0.0319 | 0.5422–0.6296 | 17.46 s |
| GRU | 5 | 0.5991 ± 0.0733 | 0.5148–0.6811 | 17.68 s |
| TCN | 5 | 0.5881 ± 0.0496 | 0.5321–0.6438 | 18.67 s |
| Vanilla LSTM | 5 | 0.5784 ± 0.0362 | 0.5403–0.6212 | 19.12 s |
| **Seq2Seq LSTM (proposed)** | 5 | **0.5614 ± 0.0638** | 0.4858–0.6597 | **20.90 s** |

**The proposed model is the worst of the seven.** Ordered by the per-window
duration error the decision layer actually consumes, the ranking is XGBoost <
persistence < Transformer < GRU < TCN < vanilla LSTM < Seq2Seq LSTM.

### The pre-registered headline test

`seq2seq_lstm` vs `persistence`, named in Task 11's docstring before the tests
ran:

| quantity | Seq2Seq | persistence | difference | p |
|---|---|---|---|---|
| per-window MAE | 20.90 s | 12.95 s | **+7.95 s (worse)** | 5.7 × 10⁻⁴⁴ |
| horizon-step accuracy | — | — | −0.0574 (worse) | 1.5 × 10⁻¹¹³ |
| STANDBY F1, seed mean | 0.5614 | 0.6815 | **−0.1200** | — |
| STANDBY F1, seed 0 only | 0.6597 | 0.6815 | −0.0218, CI [−0.0418, −0.0026] | excludes zero |

**The test resolves against the proposed model, decisively and in the opposite
direction to the one it was written to check.** Phase V was set up to ask
whether a *small positive* gap was real. There is no positive gap to test: the
untrained reference beats the proposed model by 7.95 s of per-window MAE with a
rank-biserial correlation of +0.58, and the sign was already negative in Phase
II's own single-seed table.

The F1 interval is computed on **seed 0 of each model** — which is the
*best* of Seq2Seq's five seeds. Even granting the proposed model its most
favourable draw, the interval excludes zero in persistence's favour. On the
seed mean the gap is five times wider.

**Persistence beats every one of the five neural models**, all five at
p_holm < 0.0001 and all five by margins (4.5–8.0 s of MAE) far above the 1 s
practical threshold: "real and material" in every case. The single comparison
in the table that favours a trained model is XGBoost over persistence, +1.45 s
MAE, p_holm = 0.0007 — real, material, and the only defensible claim that
training helps on this machine.

### Friedman over seeds, and how little it can see

Blocks = 5 seeds, treatments = the 6 trained models: χ² = 12.77, p = 0.0256,
Kendall's W = 0.511, CD = 3.37. XGBoost takes mean rank **1.00** — it is the
best model on every single seed block. Seq2Seq takes 5.00, the worst.

The critical difference is 3.37 on a 1–6 rank scale, so **the only pair the
diagram separates is XGBoost vs Seq2Seq** (4.00 apart). Everything else is
joined. Measured power at this design is **0.14 for a 1-sd effect and 0.53 for
2 sd** — the seed-blocked comparison is nearly blind, exactly as the n = 4
cross-machine test in 11D is, and for the same reason. The confirmatory
evidence above does not depend on it: those tests are paired over 5,449
windows, not over 5 seeds.

---

## Task 11B — the decision layer

Day-block bootstrap, 2,000 draws, on the minimum-dwell (Phase IV) labelling:
**394 held-out episodes spread over 18 factory days**, 73.6 days of span. The
resampling unit is the factory day, every policy is scored on the same
resample, and a day drawn twice is re-keyed so the two copies do not share one
day's restart budget.

Savings against the observed status quo, in USD/yr per machine:

| policy | S1 (10 min) | S2 (60 min) | S3 (240 min) |
|---|---|---|---|
| never_shutdown | −219.8 [−334.4, −131.8] | −157.1 [−275.9, −65.3] | +68.6 [−83.9, +213.3] |
| immediate | −219.7 [−338.4, −126.0] | −307.5 [−429.9, −211.9] | −623.4 [−774.2, −478.4] |
| static_break_even | −166.7 [−290.1, −71.0] | −66.0 [−195.7, +24.2] | +6.3 [−170.3, +168.2] |
| ski_rental | −46.7 [−93.0, −6.8] | +0.0 [−25.0, +28.8] | +30.4 [−78.6, +154.5] |
| **forecast_opt (proposed)** | **−28.5 [−78.7, +0.8]** | **+2.9 [−55.9, +46.2]** | **+94.5 [−58.0, +237.1]** |
| oracle | +25.5 [+14.6, +38.2] | +37.6 [+17.3, +61.2] | +160.8 [+64.7, +266.2] |

The four contrasts the paper's decision claim rests on:

| contrast | S1 | S2 | S3 |
|---|---|---|---|
| proposed − status quo | −28.5, p=0.072 | +2.9, p=0.828 | +94.5, p=0.196 |
| proposed − ski-rental | +18.2, p=0.553 | +2.8, p=0.729 | +64.1, p=0.212 |
| proposed − static heuristic | **+138.2, p<0.001** | **+68.9, p=0.001** | **+88.3, p<0.001** |
| proposed − oracle | **−54.0, p<0.001** | **−34.8, p<0.001** | **−66.3, p<0.001** |

**The proposed policy cannot be shown to beat doing nothing.** In none of the
three scenarios does its interval against the status quo exclude zero, and in
S1 the point estimate is *negative*: at a 10-minute break-even the policy
shuts down often enough that restart costs outweigh the standby it avoids.
The S2 headline the paper has been quoting is +2.9 USD/yr against an interval
of ±50 — the number is indistinguishable from zero and always was; Phase V
only supplies the interval that makes it visible.

**Nor does it beat ski-rental.** The competitive heuristic needs no forecaster,
no training and no features, and its interval overlaps the proposed policy's in
all three scenarios. Everything the forecasting stage buys over a rule that can
be written in one line is inside the noise of an 18-day sample.

**What does survive is the comparison against the static break-even rule** —
the fixed-threshold policy Phase III set out to replace. The proposed policy
beats it by +138 / +69 / +88 USD/yr with p ≤ 0.001 in every scenario, and the
margin is economically significant on the 5 USD/yr threshold. That is the
decision-layer claim the paper can make: *adaptive thresholds beat a fixed
one*, not *forecasting beats not forecasting*.

**The oracle gap is also real.** The proposed policy sits significantly below
the perfect-foresight bound in all three scenarios, so the head-room is
genuine and mostly uncaptured — 7.6% of oracle in S2, not the 55% the Phase
I–III labels reported (see *Consequent corrections* below).

**The width is the sample, not the method.** Eighteen factory days is the
binding constraint: the day-block bootstrap has eighteen units to resample, so
no amount of care in the estimator narrows these intervals. Any future claim
of economic benefit from this layer needs a longer record, and the paper should
say so rather than quoting a point estimate to one decimal place.

---

## Task 12 — repeated runs

### 12A — forecasting, 41 runs over 7 models

Seed counts are stated per model, not uniform: 15 for the tree, 5 for each
neural architecture, 1 for the deterministic reference. The step checkpoints
**per seed**, so an interruption costs one seed rather than a model.

The finding is not the spread of any one model but its structure:

| | XGBoost | the five neural models |
|---|---|---|
| F1 sd over seeds | 0.0025 | 0.032 – 0.073 |
| worst-to-best swing | 0.009 F1 | 0.081 – 0.166 F1 |

**The neural family is 13–29× less stable than the tree**, and the swing
between a good and a bad seed of the same architecture (up to 0.166 F1) is
larger than the entire spread between architectures.

**Phase II's neural numbers are all seed 0, and seed 0 is unusually kind.**
Every one of the five reproduces exactly at seed 0 — 0.6597, 0.6609, 0.6212,
0.6438, 0.6296 — so training here is bit-reproducible and Phase II's protocol
was sound. But seed 0 is the *best of five* for four of the five architectures.
Phase II did not cherry-pick; it ran one seed, as its protocol said. The
consequence is the same either way: every neural row in that table is an
optimistic draw, and the table cannot be read as a ranking.

### 12B — labelling, 10 seeds, and the finding that matters most

Re-fitting the state model at ten seeds on one deterministic preprocessing:

| | over all 10 seeds | over the 7 that pass the physics battery |
|---|---|---|
| STANDBY hours (whole record) | 84.6 ± 28.5 h | **92.4 ± 2.8 h** |
| coefficient of variation | **33.7%** | **3.0%** |
| range | 13.8 – 121.6 h | 87.3 – 94.2 h |
| STANDBY power | 4 570 ± 3 543 W | 4 676 ± 343 W |

**Three seeds in ten converge on a cluster that is not standby at all:**

| seed | physics | STANDBY | mean power | current ratio | what it actually found |
|---|---|---|---|---|---|
| 6 | 0.80 | 121.6 h | 12 961 W | 0.404 | absorbed a loaded running state |
| 8 | 0.60 | 63.2 h | 0.8 W | 0.0006 | the OFF state |
| 9 | 0.60 | 13.8 h | 1.1 W | 0.0006 | the OFF state |

A "standby" cluster at 0.8 W drawing 0.06% of rated current is the machine
switched off; one at 13 kW is the machine working. **The headline state-time
quantity of the whole paper is therefore not reproducible from the seed
alone** — EM initialisation decides it, and in 30% of initialisations the
cluster labelled STANDBY is physically a different state.

**What rescues it is the physics battery, and this is the constructive
result.** The T1–T10 checks are not decoration: they score exactly those three
seeds at 0.60–0.80 and the other seven at 1.00, and conditioning on a passing
score collapses the spread from ±33.7% to **±3.0%**. The pipeline is
reproducible *because* of the physics validation, not despite the clustering —
which is a stronger argument for the paper's own contribution than the paper
currently makes. It does, however, have to be stated as part of the method:
the labeller is fit and then **accepted or rejected** on the physics score, and
Phases I–IV never said so because they never ran a seed that failed.

Flicker is 0.00% at every seed, so the minimum-dwell constraint is unaffected
by any of this, and ARI against seed 0 stays ≥ 0.889 — the *partition* is
stable because OFF and PEAK_LOAD dominate the record; it is the identity of the
STANDBY cluster specifically that moves.

### 12C — decision layer, 10 seeds

S2_moderate, minimum-dwell labels:

| quantity | mean ± sd | range |
|---|---|---|
| proposed policy | **+2.92 ± 0.59 USD/yr** | +2.56 – +4.57 |
| % of oracle | 7.8 ± 1.6% | 6.8 – 12.1 |
| attainable head-room | 37.63 ± 0.00 USD/yr | — |
| forecaster MAE | 8 954 ± 156 s | — |
| shutdowns / min-off violations | 15 / 2 (no seed variance) | — |

Model-fitting variance is small (±0.59 USD/yr) beside the sampling variance
Task 11B found (±50 USD/yr). **The uncertainty in the economic claim is the
18-day sample, not the model.** The forecaster's own MAE is ~2.5 h on episode
durations, which is why the policy captures under 8% of the head-room.

---

## Consequent corrections to earlier phases

**None of these are applied yet.** Phase V's remit was to test Phases I–IV, not
to rewrite them; each item below states what must change and where, for the
phase that owns it.

**1. Phase II, Task 5 forecasting table — the ranking is wrong.**
Every neural row is a single seed (seed 0), and seed 0 is the best of five for
four of the five architectures. The table must be restated as mean ± sd over
the seeds in `outputs/phase5/task12/.../forecasting_seeds.json`. The order
changes: XGBoost (0.6893) and untrained persistence (0.6815) both move above
*all five* neural models (0.561–0.599). Any sentence describing the Seq2Seq
LSTM as the proposed or best forecaster no longer holds.

**2. Phase I, the novelty claim — the second of the three contributions does
not survive.** "Temporal sequence forecasting of no-load duration (Seq2Seq
LSTM)" is the component Phase V finds to be worse than not forecasting. The
honest version of the framework claim keeps the physics-validated state
inference and the optimisation layer, and either drops the sequence model or
demotes it to a compared baseline. A gradient-boosted tree over the same
windows is the component that works.

**3. Phase III, `paper/decision_layer.tex` — the minimum-dwell figures are not
reproducible.** The text states +7.2 USD/yr at 16.8% of oracle with head-room
falling to 42.9 USD/yr. Ten seeds of the identical configuration give
**+2.92 ± 0.59 USD/yr at 7.8 ± 1.6% of oracle**, on head-room of **37.63
USD/yr with no seed variance at all**. Head-room is a property of the episodes
and the coefficients, so a seed cannot explain the 42.9 → 37.63 discrepancy;
something else differs between that run and this one, and Phase III has to
find it. Both quoted numbers (+38.6 and +7.2) need re-deriving before either
is used again. The payback paragraph inherits the correction.

**4. Every state-time total in the paper needs two changes.** They must be
quoted (a) conditional on the physics battery passing, and (b) to two
significant figures at most. Task 11C already showed the sampling interval is
±15–25%; Task 12B now adds that the labelling itself moves the answer by
±33.7% unconditionally, or ±3.0% conditionally. Numbers written to three
significant figures assert a precision that neither interval supports.

**5. The method description is incomplete in a way that matters for
reproducibility.** Phases I–IV describe fitting the GMM and validating it.
They do not say that a fit which fails the physics battery is *rejected and
refit*, because no failing seed was ever run. Since 3 seeds in 10 fail, the
acceptance step is load-bearing and belongs in the method, not in the
validation appendix. This is a strengthening of the paper's own contribution,
but only if it is stated.

**6. Phase IV is unaffected.** Its cross-machine claims were tested directly in
Task 11D: the transfer degradation survives (p = 0.0078, d = +1.75), and the
motor-machine comparison was already reported as untestable at n = 4.
