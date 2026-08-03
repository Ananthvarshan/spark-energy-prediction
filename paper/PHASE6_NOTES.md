# Phase VI — Explainability and ablation: completion notes

Covers Task 13A (Shapley attribution of both forecasters), 13B (the state
layer), 13C (transition structure), 13D (publication figures) and 13E (the
component ablation). Every number below is from runs on this machine.

Reproduce with `python run_phase6.py` (11.6 min measured end to end;
`--skip task13a` cuts it to under 4). Every step is deterministic — a clean
re-run of the whole phase
reproduces the attributions below digit for digit, including the refitted
window forecaster's F1 to sixteen significant figures.

## Files produced

| File | Role |
|---|---|
| `experiments/task13a_forecaster_shap.py` | Task 13A — decision forecaster, chance-constraint classifier, window forecaster |
| `experiments/task13b_state_shap.py` | Task 13B — state surrogate, two targets, compared against Phase II's ranking |
| `experiments/task13c_transitions.py` | Task 13C — learned / realised / post-dwell transition matrices, eight machines |
| `experiments/task13e_ablation.py` | Task 13E — the ablation table, CSV and LaTeX |
| `experiments/make_figures_phase6.py` | twelve figures (seven new, five Phase V re-renders) |
| `run_phase6.py` | master runner |
| `paper/explainability.tex` | §7 draft |
| `paper/ablation.tex` | §8 draft |
| `outputs/phase6/**` | attribution tables, raw Shapley arrays, matrices, ablation, figures, logs |

`experiments/task5_forecasting_baselines.py` gained two things: a
`xgb_design_feature_names` helper that names the 217 design columns next to the
featuriser that builds them, and an optional `capture` argument to
`train_xgboost` that hands back the fitted booster. Both are additive; the
Task 5 and Task 12 call sites are unchanged.

---

## One thing had to be settled before any attribution was computed

**There are two XGBoost forecasters in this framework and they are not the same
model.** The improvement plan's Task 13A refers to "the XGBoost forecaster",
but Phase IV's 94.5% calendar claim is about the decision layer's
remaining-duration model (10 features, one row per decision epoch), while Phase
V's promotion of a tree to the paper's forecasting component is about the
window state model (217 design columns, one row per window × horizon step).
They serve different consumers, are scored on different units, and — as it
turns out — use almost disjoint information. Both are attributed, separately,
and the ablation table refuses to put them in the same column.

**The window model is refitted here, and the refit is verified rather than
assumed.** Task 5 and Task 12 store predictions, not boosters, so the model has
to be rebuilt to be explained. The first attempt used Task 5's module default
(`DECIMATE = 5`) and produced STANDBY F1 0.7035 against the stored seed-0
0.6852 — a *better* model, and the wrong one, because Phase V's table is the
Task 12A run at `DECIMATE = 10`. The step now imports Task 12A's constants and
**raises rather than continuing** if seed 0 does not reproduce the stored
metrics to 1e-6. It reproduces them exactly: F1 0.6852131546894031, MAE
11.62598641952652 s.

**Shapley additivity is checked here rather than by the library, and the
tolerance is stated.** `shap`'s built-in assertion compares against the model
output at an *absolute* tolerance of 1e-2. The duration forecaster predicts
seconds and its outputs run to 1.4e5, and the tree traversal accumulates in
float32, so its worst reconstruction error — 0.04 s on a prediction of 2e4 s,
a relative error of 2e-6 — trips a threshold written for models whose outputs
are probabilities, and the whole attribution aborts. The check is now performed
on the model's raw margin and on the relative scale, and the measured value is
recorded in the results JSON rather than the check being switched off. Largest
relative error over the four forecaster attributions: **1.97e-6**, four orders
of magnitude below the smallest difference any claim here rests on.

---

## Task 13A — the two forecasters

### The decision forecaster: the 94.5% figure does not survive, the conclusion does

Phase IV reported that four calendar features carry 94.5% of the split gain,
measured on Phase III's fit, i.e. on the Phase I–III labelling. That reproduces
exactly: the same four features sum to **94.4%** of the gain here on the same
labelling. Shapley attribution of the identical model gives a different number:

| feature group | SHAP / gain, Phase I–III labels | SHAP / gain, minimum-dwell labels |
|---|---|---|
| factory calendar (7 features) | **78.0%** / 96.2% | **82.4%** / 96.7% |
| machine history (2) | 18.8% / 2.7% | 13.3% / 2.3% |
| elapsed idle time (1) | 3.2% / 1.2% | 4.3% / 1.0% |

Both labellings are run because the answer could have depended on which one is
used: the minimum-dwell constraint merges 3,224 idle episodes into 1,312 and
raises the median from 12 s to 90 s, so the forecaster is fitted on a
materially different problem. It does not depend on it. The gain overstates the
calendar share by 14–18 points either way.

The two measures agree only at Spearman ρ = 0.72 (min-dwell labels; ρ = 0.43 on
the Phase I–III ones), and the disagreement is structural rather than noise
(min-dwell labels):

| feature | gain | SHAP | rank by gain | rank by SHAP |
|---|---|---|---|---|
| `cur_is_weekend` | **63.2%** | **5.1%** | 1 | 5 |
| `onset_hour` | 17.5% | **43.1%** | 2 | 1 |
| `prev_productive_log` | 2.2% | 11.3% | 4 | 3 |

`cur_is_weekend` is a binary split near the root: it is split on early, on a
partition that later splits refine, and accumulates gain out of all proportion
to the change it makes to any individual prediction. This is the textbook
failure mode of gain-based importance, and the paper had a claim resting on it.

**What changes: the number, from 94.5% to 82%. What does not change: the
conclusion.** A model that is 82% plant calendar still transfers between any
two machines on the same shift pattern and to nothing else, which is what Task
9's transfer matrices measured independently. What the attribution adds is that
the machine's own production history is not negligible — 13.3% against gain's
2.3% — so "the forecaster is a clock" is too strong; "the forecaster is four
fifths clock" is right.

### The chance-constraint classifier keys on something else, which is why it is a separate model

| | duration regressor | feasibility classifier |
|---|---|---|
| elapsed idle time | 4.3% (rank 6) | **33.1% (rank 1)** |
| factory calendar | 82.4% | 54.5% |
| machine history | 13.3% | 12.3% |

Phase III argued that a point forecast of E[R] carries no information about
P(R ≥ h_min) on a distribution this skewed, and fitted the constraint its own
model on that reasoning. The attribution confirms the reasoning empirically:
the two models weight the same ten features differently, and the feature that
moves most is exactly the one the survival question depends on.

### The window forecaster: an electrical model, and largely a persistence one

| feature group | SHAP share | gain share |
|---|---|---|
| raw electrical | **40.8%** | 24.2% |
| rolling statistics | 23.9% | 20.9% |
| derived electrical | 20.1% | 36.7% |
| temporal / calendar | **11.6%** | 18.1% |
| horizon step index | 3.7% | 0.1% |

**11.6% calendar here against 82.4% in the decision forecaster.** Phase IV
inferred from the transfer results that the two layers depend on almost
disjoint information; this is that inference measured directly on one machine
rather than inferred from a transfer matrix.

Two results in this table are inconvenient and are reported as they stand.

**Voltage is the largest single contributor** — 20.0% across its nine design
columns, 11.8% on `voltage|last` alone, against 3.0% of the split gain. Supply
voltage is not a state variable of the machine; it sags under load and recovers
when the drive unloads, so it is a real but indirect indicator and one tied to
this site's point of common coupling. It is the contribution least likely to
transfer, and no earlier phase noticed it because gain ranks it tenth of the
twenty-five channels.

**Grouping the 217 columns by what they summarise explains Phase V's
persistence result:**

| summary | SHAP share |
|---|---|
| final observed value of the lookback | **43.5%** |
| block means (4 sub-blocks × 24 channels) | 26.4% |
| block standard deviations | 26.3% |
| horizon step index | 3.7% |

A model whose largest single input is the state the machine is in *now* is a
refinement of "nothing changes", and cannot beat that reference by much. It
does not: the measured margin over untrained persistence is 1.45 s of
per-window MAE. Phase V established that gap statistically; this says why it is
small.

---

## Task 13B — the state layer

Surrogate fitted on the minimum-dwell labels, 4.33 M clean rows, two targets.
The informative one is STANDBY against {WORKING, PEAK_LOAD} on energised rows
only — separating OFF from everything else is trivial and would otherwise
dominate.

| channel | SHAP share | gain share | MI rank |
|---|---|---|---|
| `current` | **17.8%** | 28.9% | 4 |
| `power_factor` | **16.9%** | 5.1% | 2 |
| `active_power` | 12.3% | **60.3%** | 1 |
| `apparent_power` | 9.5% | 0.8% | 6 |
| `q_p_ratio` | 7.9% | 2.9% | 3 |

(Mutual information separates these five by less than 0.001 nats and its
ranking should not be read as an ordering; it is reported because agreement
between three measures with different failure modes is what makes the
attribution worth quoting, and MI puts all five inside its top six of 24.)

**By split gain, `active_power` takes 60.3% and `power_factor` 5.1%; by Shapley
attribution the order reverses.** Power magnitude is what the trees split on;
power factor is what changes the predictions. That is the paper's physics
argument confirmed by the data rather than asserted from motor theory — and it
is the same gain-versus-Shapley correction that Task 13A found, appearing
independently in a different model on a different target.

The four physics-named channels carry **30.9%** between them, with
`power_factor` at rank 2. That is a real result but not a dominant one: the raw
electrical channels still carry 46%, and the paper should not claim more than
"power factor is the strongest single discriminator after current".

**Agreement with Phase II's Task 3 ranking on the earlier labelling: Spearman
ρ = 0.833 (p = 4×10⁻⁷), four of five top features shared.** The minimum-dwell
re-labelling does not move the explanation, which is what Phase V's correction
4 needed in order to leave the explainability claim standing.

**The surrogate's 99.74% accuracy is not evidence of anything.** Its targets
are the state model's own labels. It measures that a tree can re-express a
mixture model's decision boundary, which was never in doubt. Phase II's notes
already recorded this; it is repeated here because the number is large enough
to be quoted by accident.

---

## Task 13C — transition structure

Three matrices per machine — the HMM's learned `transmat_`, the frequencies
realised by Viterbi decoding (`state_raw`), and the frequencies after the
minimum-dwell constraint (`state`) — with transitions never counted across an
acquisition gap.

On pelletizer-I:

| | learned | realised | after min-dwell |
|---|---|---|---|
| OFF → PEAK_LOAD | **0** | **0 events** | **0 events** |
| OFF → WORKING | 1.1e-4 | 129 events | 126 events |
| WORKING → OFF | 9.2e-4 | 124 events | **25 events** |
| PEAK_LOAD → OFF | 0 | 0 events | **3 events** |
| expected dwell, STANDBY | 82 s | 96 s | **246 s** |
| expected dwell, WORKING | 24 s | 25 s | 62 s |

**The claim the paper makes holds, and holds exactly.** OFF → PEAK_LOAD is zero
in the learned matrix and occurs zero times in 5.47 million readings. The 126
OFF → WORKING events are 6.1e-5 of the outgoing mass from OFF and are what a
1 Hz sampler produces when a machine crosses the standby band faster than one
second — not a violation of the ordering.

**The minimum-dwell constraint acts on the transitions, not only on the
labels**: WORKING → OFF falls from 124 events to 25 and the expected STANDBY
dwell rises from 96 s to 246 s.

**It also introduces an artefact, which is stated rather than trimmed.** Three
PEAK_LOAD → OFF transitions appear *after* the constraint that the decoder
never produced: absorbing a sub-threshold run into its neighbours can join two
states that were not adjacent. Three events in 5.47 M rows changes nothing that
is reported, but it is a property of the operator, not of the machine, and no
one had looked before.

Across the plant:

| machine | min self-transition (realised) | STANDBY dwell | largest forbidden transition |
|---|---|---|---|
| pelletizer-I | 0.984 | 246 s | 1.9e-4 |
| pelletizer-II | 0.995 | 297 s | 2e-6 |
| milling-I | 0.964 | 111 s | 5e-6 |
| milling-II | 0.935 | 106 s | 3.9e-5 |
| exhaust-fan-I | 0.996 | 6116 s | **3.4e-3** |
| exhaust-fan-II | 0.957 | 23 s | 4.8e-4 |
| dpc-I | 0.993 | 147 s | 3.6e-4 |
| dpc-II | 0.992 | 122 s | 1.6e-4 |

**The transition structure fails on exactly the machine Phase IV said it should
fail on**, and by the largest margin in the table: exhaust-fan-I, whose state
names Task 8 showed to be mis-assigned because a near-zero-power cluster
carrying standing reactive current outranks the genuine off state. That is an
independent measurement arriving at the same conclusion, and it is worth a
sentence in the paper for that reason. Exhaust-fan-II's 23-second STANDBY dwell
is the other visible symptom: a state the machine does not really occupy has no
characteristic duration.

---

## Task 13E — the ablation

| stage | configuration | metric | value |
|---|---|---|---|
| state | GMM only (i.i.d.) | flicker | 66.9% |
| state | + HMM (Viterbi) | flicker | 44.5% |
| state | + minimum dwell | flicker | **0.0%** |
| forecast | no forecaster (persistence) | F1 / MAE | 0.6815 / 12.95 s |
| forecast | + Seq2Seq LSTM | F1 / MAE | 0.5614 ± 0.0638 / 20.90 s |
| forecast | + boosted tree | F1 / MAE | **0.6893 ± 0.0025 / 11.50 s** |
| decision | static break-even | USD/yr | −66.0 [−195.7, +24.2] |
| decision | ski-rental | USD/yr | +0.0 [−25.0, +28.8] |
| decision | + wrong estimand | USD/yr | −28.6 [−155.2, +49.3] |
| decision | + no chance constraint | USD/yr | −15.0 [−62.6, +25.6] |
| decision | + chance constraint (proposed) | USD/yr | **+2.9 [−55.9, +46.2]** |
| bound | oracle | USD/yr | +37.6 [+17.3, +61.2] |

Every cell is read from a stored Phase II–V output; nothing is refitted.

**Per-component verdicts, derived from the table rather than asserted:**

| component | contribution | verdict |
|---|---|---|
| HMM | flicker 66.9% → 44.5%, physics unchanged | contributes, partially — a prior cannot remove flicker |
| minimum dwell | flicker 44.5% → 0.0%, median dwell 13 s → 92.5 s | contributes; it is what removes flicker |
| Seq2Seq LSTM | −0.120 F1, +7.95 s MAE against persistence | **does not contribute** |
| boosted tree | +0.008 F1, −1.45 s MAE, p_holm = 7e-4 | contributes; the only trained forecaster that beats not training |
| mean estimand | +31.5 USD/yr over the log-median variant | **largest single decision-layer effect** |
| chance constraint | +17.8 USD/yr; interval still spans zero | contributes to the point estimate only |
| optimisation vs static rule | +68.9 USD/yr, p = 0.001 | contributes; the one significant decision-layer contrast |

**Three points the table settles that prose had left open.**

The HMM row and the min-dwell row carry the *same* physics score (5/5) and
schedule score to within 0.06 points. This is the expected result, not a null
one: the dwell constraint changes when the machine is said to change state, not
which cluster is standby. A table showing it improving the physics compliance
would indicate a leak between the two.

Targeting the conditional mean rather than the conditional median is worth
31.5 USD/yr — more than the chance constraint (17.8) and more than the entire
gap between the proposed policy and the status quo (2.9). Phase III argued this
from the shape of the duration distribution; the ablation prices it.

Every policy's interval against the status quo spans zero except the oracle's,
so **no row of the decision block, read on its own, demonstrates a saving**.
The paired contrasts are a different question and one of them survives:
proposed minus static break-even is +68.9 USD/yr with an interval excluding
zero (p = 0.001), because both policies are scored on the same bootstrap
resample and the pairing removes the day-to-day variance that makes the
individual intervals so wide. The distinction is the whole reason Task 11B
scores every policy on the same draw, and the ablation table's layout keeps
both readings visible instead of quoting whichever is more favourable.

---

## Task 13D — figures

Twelve figures under `outputs/phase6/figures/`, each as PDF and PNG:

| figure | shows |
|---|---|
| `fig_task13a_decision_shap_{phase4,auto}` | beeswarm + Shapley-against-gain, both labellings |
| `fig_task13a_window_shap` | window model by channel and by window summary |
| `fig_task13b_state_shap_phase4` | STANDBY-vs-productive beeswarm + three importance measures |
| `fig_task13c_transition_matrix` | the three 4×4 matrices, log colour, event counts annotated |
| `fig_task13c_transition_graph` | the same as a state graph with expected dwells |
| `fig_task13e_ablation` | the ablation, three panels for three units |
| `fig_p6_cd_diagram` | Phase V critical-difference diagram |
| `fig_p6_decision_forest` | Phase V decision intervals, three scenarios |
| `fig_p6_power_curve` | Phase V power against n, with the observed d marked |
| `fig_p6_standby_hours` | eight machines, weekly-block intervals |
| `fig_p6_seed_stability` | F1 by seed for all seven forecasters |

What "publication quality" changed, beyond resolution: figures are sized to
IEEE single (3.5 in) and double (7.16 in) column widths so they are placed at
100% and 8 pt type stays 8 pt; rendered suptitles are removed because they
duplicate a caption that cannot be edited in proof; fonts are embedded
(`pdf.fonttype 42`); and no result is encoded by hue alone — an interval that
crosses zero is grey *and* hollow, and a machine with an unusable interval is
hatched, so both survive a monochrome printer.

Three figures gained content in the re-render rather than only styling.
`fig_p6_standby_hours` now marks which machines pass the physics battery and
prints the number of weekly blocks inside each bar, so the two milling machines'
5.5 h/day — the largest figure among the motor-driven machines, resting on two
blocks — cannot be read as the largest opportunity. `fig_p6_seed_stability` is
new; Phase V had the data but no figure. `fig_p6_power_curve` marks the effect
actually observed on the four motor machines (d = 0.56) against the n = 4
curve, so the reader can read its power off the axis instead of taking 0.00 on
trust.

---

## Consequent corrections to earlier phases

**None of these are applied yet**, in keeping with the convention Phase V used:
each states what must change and where, for the phase that owns it.

**1. Phase IV, Task 9 Part A — the 94.5% figure must be restated.** The
sentence "the four calendar features carry 94.5% of the gain" is a gain-based
importance and gain is demonstrably misranking this model: `cur_is_weekend`
takes 63.2% of the gain and 5.1% of the Shapley attribution. On the labelling
that sentence was measured on, the honest statement is **78.0% of the
attribution (96.2% of the gain)**; under the minimum-dwell labelling the paper
now quotes, **82.4% (96.7%)**. Either way the machine's own production history
carries 13–19% rather than the 2.4% the gain implies. The *conclusion* drawn
from the figure — that the model is carrying the plant calendar and will
therefore transfer within the facility and nowhere else — is unaffected and is
now better supported.

**2. `paper/related_work.tex` §2.1 still needs the sentence Phase IV asked
for**, recording that at n > 10⁶ both BIC and silhouette keep buying
components. Phase IV left it "for the next phase that touches it"; Phase VI has
not touched that file either, and it is still outstanding.

**3. Phase II, Task 3 — the multiclass target's headline should be dropped, not
updated.** Task 13B reproduces it (`active_power` first, raw electrical 50.6%),
but the target is dominated by the OFF boundary, which power magnitude alone
resolves. Only the STANDBY-vs-productive target says anything about the paper's
claim, and quoting both invites the reader to average them.

**4. A limitation the paper does not currently state.** The window forecaster
draws 20% of its attribution from supply voltage, which is a property of the
site's electrical installation rather than of the machine. Any claim about
deploying that model elsewhere should carry the caveat, and the cross-dataset
failure Task 10 measured is consistent with it.

**5. The minimum-dwell operator has a small artefact worth one sentence in the
method.** It can create a transition between two states that were not adjacent
in the decoded sequence: three PEAK_LOAD → OFF events in 5.47 M rows. Nothing
reported changes, but the operator is described in the paper as removing runs,
and it also — rarely — joins states.

---

## Things a reviewer will ask that are not yet answered

- **The attributions are for one machine.** Task 13C covers all eight; Tasks
  13A and 13B cover pelletizer-I only. Whether power factor leads the STANDBY
  split on pelletizer-II and the milling machines is a cheap experiment that
  has not been run, and it is the obvious strengthening of the physics claim.
- **SHAP on a surrogate is not SHAP on the model.** The state layer is a
  GMM-HMM; what is attributed is a tree fitted to its output. The rankings are
  meaningful, the mechanism is one step removed, and no attribution method for
  a Viterbi-decoded mixture model is applied here.
- **Feature dependence is not modelled.** Tree SHAP with the path-dependent
  perturbation splits credit between correlated features in a way that depends
  on the tree structure, and `active_power`, `apparent_power` and `current` are
  strongly correlated on this record. The ordering among *those three*
  specifically should be treated as less firm than the gap between them and the
  calendar or rolling-statistic groups.
- **The ablation is single-machine and single-scenario.** The decision block is
  S2 on pelletizer-I. Task 13E takes `--scenario`, and S1 and S3 are stored, so
  the other two blocks exist; they are not in the paper table.
- **No ablation of the physics battery itself.** Phase V showed that
  conditioning on a passing physics score collapses the labelling spread from
  ±33.7% to ±3.0%. That is the strongest accept/reject result in the project
  and it is not a row in the ablation table, because it is measured over
  labelling seeds rather than over the held-out blocks the rest of the table
  uses.
