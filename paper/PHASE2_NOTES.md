# Phase II — Methodology: completion notes

Covers Task 3 (feature engineering + importance), Task 4 (state-detection
comparison) and Task 5 (forecasting baselines). All numbers below are from
runs on this machine against `outputs/imdeld_labelled/pelletizer-I_labelled.csv`
(5,474,431 rows @ 1 Hz, 2017-10-30 → 2018-04-03, 155 days).

## Files produced

| File | Role |
|---|---|
| `src/features.py` | Task 3 feature engineering — 24 features in 4 groups |
| `experiments/common.py` | shared loading, block sampling, ground-truth-free metrics |
| `experiments/task3_feature_importance.py` | Task 3 |
| `experiments/task4_state_detection.py` | Task 4 main comparison |
| `experiments/task4_sticky_prior_sweep.py` | Task 4b — prior-strength diagnostic |
| `experiments/task4_flicker_diagnosis.py` | Task 4c — why the prior can't work |
| `experiments/baseline_models.py` | Task 5 model zoo (5 neural architectures) |
| `experiments/task5_forecasting_baselines.py` | Task 5 |
| `experiments/make_figures.py` | result figures |
| `outputs/phase2/**` | all tables, JSON, labels, figures, logs |

New dependencies installed: `hmmlearn`, `xgboost`, `shap`.

---

## Task 3 — feature engineering and importance

24 features in four groups: raw electrical (5), derived electrical (4 —
`power_factor`, `q_p_ratio`, `load_factor`, `s_residual`), rolling statistics
(8, 300 s trailing window), temporal (7). Rolling windows are **trailing, not
centred**, and are computed within contiguous segments so none spans the
record's 30-day acquisition gap.

Three importance measures on two targets. The paper-relevant target is
**STANDBY vs productive on energised rows only** — separating OFF is trivial,
separating STANDBY from a lightly-loaded WORKING state is the actual problem.

> **Correction applied in Phase VII (from PHASE6_NOTES correction 3).** The
> four-class target was also run and its group importances are stored
> (`importance_by_group_multiclass.csv`: raw electrical 0.597 SHAP, derived
> 0.243). **That headline is dropped from the paper rather than updated.**
> Task 13B reproduces it — `active_power` first, raw electrical 50.6% — but
> the four-class split is dominated by the OFF boundary, which power
> magnitude alone resolves, so the ranking it produces says nothing about the
> claim the paper makes. Only the STANDBY-vs-productive target speaks to that
> claim, and quoting both invites the reader to average two numbers that
> answer different questions. The stored CSV is retained for audit; no
> section of the paper cites it.

Top features by consensus rank: `active_power`, **`power_factor`**,
**`q_p_ratio`**, `current`.

| group | mutual info | XGBoost gain | SHAP |
|---|---|---|---|
| raw electrical | 0.350 | 0.560 | 0.537 |
| derived electrical | 0.250 | 0.438 | 0.310 |
| rolling statistics | 0.388 | 0.001 | 0.116 |
| temporal | 0.012 | 0.002 | 0.036 |

Two findings worth putting in the paper:

**The physics argument is empirically confirmed.** `power_factor` and
`q_p_ratio` rank #2 and #3, and derived electrical features carry 31–44% of
total importance from four features. Methods 1 and 2 of the five-method proof
were previously justified by motor theory alone; they now have a measured
basis.

**Temporal features are nearly irrelevant to this split** (1–4%). The
STANDBY/productive distinction is being made on physics, not on shift
schedule. That is the desirable outcome — a model leaning on `hour_of_day`
would not transfer to a facility with a different schedule.

**Caveat that must appear in the paper.** The surrogate reaches 99.92%
accuracy, but the targets *are* the GMM-HMM's own labels, and the GMM-HMM was
fitted on a subset of these same features. The surrogate is therefore largely
re-deriving a deterministic function of its own inputs. **The rankings are
meaningful; the accuracy is not evidence of anything** and should not be
reported as validation. This is an explainability result. Validation is
Task 4's job. The module docstring says so; make sure the paper does too.

The high mutual information but near-zero gain for rolling statistics is the
classic redundancy signature: MI is univariate, and a rolling mean is
individually informative but adds nothing once the tree already has the
instantaneous value.

---

## Task 4 — state detection comparison

Protocol: 64 contiguous 2-hour blocks spread across the record, never crossing
a gap; alternate blocks form FIT (32 blocks, 230,400 rows) and EVAL (32 blocks,
230,400 rows). All five methods use the same log1p+standardised 6-channel
representation as `validate_gmm.build_feature_matrix`. DBSCAN and spectral
clustering are transductive, so their labels are extended to EVAL by
1-nearest-neighbour — applied identically to both.

| method | flicker % | flicker STANDBY-binary % | median dwell s | physics | schedule % | STANDBY h |
|---|---|---|---|---|---|---|
| K-Means | 78.23 | 49.09 | 2 | 1.00 | 84.4 | 4.38 |
| DBSCAN (28 clusters) | 62.54 | 50.30 | 4 | 1.00 | 84.4 | 4.34 |
| Spectral | 98.18 | 98.82 | 2 | 0.80 | 89.0 | 15.01 |
| GMM (i.i.d.) | 66.94 | 53.55 | 3 | 1.00 | 84.5 | 4.17 |
| GMM-HMM | 44.47 | 44.70 | 13 | 1.00 | 84.5 | 4.16 |
| **GMM-HMM + min-dwell** | **0.00** | **0.00** | **92.5** | **1.00** | **84.5** | **4.20** |
| *reference export* | *44.56* | *45.33* | *12* | *1.00* | *84.5* | *4.18* |

**Validity check:** the re-implemented GMM-HMM reproduces the pipeline's own
exported labels almost exactly (44.47 vs 44.56% flicker; 4.16 vs 4.18 STANDBY
hours). The harness is measuring the right thing.

**The emission model barely matters; the decoding rule does.** K-Means, GMM
and DBSCAN all pass 5/5 physics checks, agree with the factory schedule to
within 0.2 points, and find 4.17–4.38 STANDBY hours. Their pooled flicker
spans only 62–78%. Switching from i.i.d. assignment to Viterbi decoding over
the *same* emissions cuts flicker from 66.9% to 44.5% and quadruples median
dwell (3 s → 13 s); adding the minimum-dwell constraint takes it to zero. For
this problem, choosing a better clustering algorithm buys almost nothing, and
choosing a better *sequence* model buys everything — which is the argument the
paper needs Table 4 to make.

**Spectral clustering is the one genuine failure.** It labels 15.01 h STANDBY
(3.6× every other method) with 98.8% binary flicker and a no-load current
ratio of 0.001 against an acceptance band of [0.25, 0.50]. Note it scores
*highest* on schedule agreement (89.0%): a partition that over-assigns
non-productive states automatically looks good on the closed window. That is a
weakness of the schedule metric in isolation, and it is why panel D of the
figure reports STANDBY hours alongside — no single metric here should be
quoted on its own.

### Four protocol corrections made during Task 4

These are worth recording because each changed a headline number.

**1. Spike rows must be kept for this task.** `lstm_pipeline.py` drops
MAD-flagged spikes (19.5% of rows) and Task 3 follows it. Doing the same here
was wrong: removing a fifth of a 1 Hz series makes retained rows non-uniformly
spaced, so a run of *N* rows no longer spans *N* seconds. Every dwell was
silently compressed and every flicker rate inflated. Fixed; `--drop-spikes`
reproduces the old variant as a sensitivity check.

**2. The HMM must be warm-started from the GMM.** Fitting HMM emissions from
scratch by Baum–Welch converged to a materially worse local optimum than the
GMM's — its "STANDBY" absorbed part of the productive range (mean 10.3 kW, sd
15.4 kW) instead of finding the true no-load band at 3.67 kW, and it failed
the variance-ordering check. Comparing that against the GMM would have
confounded the value of temporal decoding with EM initialisation luck.
Emissions are now warm-started from the fitted GMM and held fixed
(`params="t"`, only transitions learned), making `gmm` and `gmm_hmm` a clean
ablation pair differing *only* in the decoding rule.

**3. Flicker needs a decision-relevant view.** The pooled flicker rate is
dominated by churn on the WORKING↔PEAK_LOAD boundary, which is an artefact of
splitting a continuous productive range at k=4 and is irrelevant downstream —
the shutdown rule only asks "idle or not, and for how long". The metric now
also reports flicker on the binarised STANDBY-vs-rest sequence.

**4. Methods that over-segment need a fair cluster→state mapping.** DBSCAN
picks its own cluster count and returned 28. The original mapping ranked
clusters by mean power and handed the four canonical state names to the four
lowest-power clusters, which put all four "states" inside the machine's
low-power range: DBSCAN scored PF separation 0.010, 0.06 STANDBY hours, and
98.9% flicker — an apparent total failure that was **entirely an artefact of
my mapping, not of DBSCAN**. Cluster means are now grouped into four
size-weighted power bands by 1-D k-means, so sub-clusters of one physical
state are re-merged. Corrected, DBSCAN passes 5/5 physics checks with 4.34
STANDBY hours and 62.5% flicker. Had this not been caught, the paper would
have reported a baseline failure that does not exist.

---

## Task 4b/4c — the flicker result, and why it is not what the plan assumed

The improvement plan states HMM smoothing takes flicker from 45% to <5%. That
is not what happens, and the reason is worth a paragraph in the paper.

**The sticky prior in `validate_gmm.py` is numerically inert.**
`fit_gaussian_hmm` sets `kappa = target_dwell_s / sample_interval_s` = 30
Dirichlet pseudo-counts. There are ~576,000 observed transitions. The sweep
confirms it: `fixed_kappa_30` and `sticky_scale=0` (a flat prior) give
flicker 64.09% vs 64.10% and minimum self-transition 0.9678 vs 0.9677. The
prior is present in the code, documented in comments, cited to Fox et al.
(2011), and has no measurable effect.

**Strengthening it does not fix the problem either.** Scaling the prior to 10×
the total evidence (κ = 575,920; minimum self-transition 0.9994; implied mean
dwell 1,605 s) still leaves flicker at 50.2%.

**Why:** in a Viterbi decode the cost of switching state is
log(p_self / p_switch) — measured here at **9.23 nats maximum**, and bounded
above however strong the prior. The emission log-likelihood gap between the
best and second-best state has median **27.8 nats** and 90th percentile
**41,701 nats**. At **55.7%** of timesteps the emission term exceeds the
largest penalty the transition matrix can apply, so the transition model is
arithmetically incapable of changing the decision there. The HMM degenerates
toward per-sample argmax.

**What actually fixes it:** an explicit minimum-dwell constraint at decode
time (the discrete analogue of a hidden semi-Markov duration model). At a
10 s minimum — set equal to the flicker threshold itself, so it removes
exactly what the metric defines as implausible and nothing more:

| min dwell | flicker % | median dwell s | physics | STANDBY h |
|---|---|---|---|---|
| 0 s | 64.09 | 4 | 0.80 | 5.34 |
| 10 s | 0.00 | 153 | 0.80 | 5.34 |
| 30 s | 0.00 | 327 | 0.80 | 5.37 |
| 60 s | 0.00 | 436 | 0.80 | 5.40 |

Flicker goes to zero at **no cost** to PF separation, current ratio, schedule
agreement, or the STANDBY hour total (5.34 → 5.34 h). The constraint is
removing noise, not reshaping the partition.

This also explains where the pipeline's existing low flicker really comes
from: `smooth_short_standby_labels()` in `lstm_pipeline.py` is a post-hoc
minimum-dwell filter. The HMM was getting the credit for what that function
does.

### Consequent corrections to Phase I

Two Phase I claims were contradicted by this evidence and have been rewritten:

- `novelty_and_contributions.tex`, contribution 1 — previously claimed
  "jointly learning emissions and transitions… reducing the flicker rate by an
  order of magnitude". Both halves were wrong: joint emission fitting is
  *worse* than warm-starting, and the HMM alone does not deliver an
  order of magnitude. Now states the bounded-transition-cost result and
  credits the minimum-dwell constraint.
- `related_work.tex` §2.1 — softened "HMMs resolve this" to "address", and
  added a paragraph stating precisely what a transition prior can and cannot
  guarantee, pointing to hidden semi-Markov models (`yu2010hsmm`, added to
  the bibliography) for duration modelling.

---

## Task 5 — forecasting baselines

Six models under one protocol: 10× decimation (1 Hz → 0.1 Hz), 600 s lookback,
300 s horizon, 120 s stride, chronological 70/15/15 split with windows built
inside each split, identical balanced class weights, one seed each. Test set
5,449 windows; mean true STANDBY 20.1 s of a 300 s horizon, and only 13.5% of
windows contain any STANDBY at all.

| model | accuracy | macro F1 | **F1 (STANDBY)** | STANDBY MAE (s) | RMSE (s) | params | train (s) |
|---|---|---|---|---|---|---|---|
| XGBoost (sliding window) | 0.933 | 0.753 | **0.685** | **11.63** | 42.99 | 236 trees | 250 |
| **Persistence (no training)** | 0.921 | 0.734 | **0.682** | **12.95** | 49.42 | **0** | **0** |
| GRU encoder-decoder | 0.910 | 0.731 | 0.661 | 13.70 | 49.96 | 121,604 | 502 |
| Seq2Seq LSTM (proposed) | 0.894 | 0.719 | 0.660 | 13.76 | 48.82 | 161,028 | 774 |
| Temporal CNN (TCN) | 0.905 | 0.720 | 0.644 | 15.03 | 54.46 | 131,332 | 524 |
| Transformer (PatchTST) | 0.901 | 0.721 | 0.630 | 14.76 | 49.10 | 310,488 | 324 |
| Vanilla LSTM | 0.895 | 0.714 | 0.621 | 15.60 | 50.34 | 93,816 | 385 |

**The result that matters is the second row, and it was added in Phase V.** The
original Task 5 compared six *trained* models against each other and never
against not training at all. A persistence forecast — the horizon repeats the
last observed state, no parameters, no fitting — reaches F1 0.682 and MAE
12.95 s on these same windows. It beats every neural model in the table,
including the proposed Seq2Seq LSTM (0.660 / 13.76 s), and is within 0.004 F1
of XGBoost.

Three things follow, and the paper must carry all three:

1. **The sequence-forecasting contribution as originally framed does not
   survive.** No neural architecture in this comparison earns its training cost
   on this machine; the proposed model is 0.022 F1 *below* an untrained
   reference while costing 774 s of training and 161k parameters.
2. **The ranking among trained models is unaffected.** XGBoost > GRU ≈ Seq2Seq
   LSTM > TCN > Transformer > vanilla LSTM stands as reported. What changes is
   the baseline the whole column is read against.
3. **Whether XGBoost's 0.004 F1 edge over persistence is real is a question for
   Phase V**, not for this table — see `PHASE5_NOTES.md`, which tests it.

The horizon is short (300 s) relative to state dwell times (median 71 s but
heavy-tailed, with STANDBY runs lasting minutes to hours), so "nothing changes
in the next five minutes" is right most of the time. That is not an artefact to
be explained away: it is the operating regime, and a forecaster deployed here
has to beat it.

Reproduce the persistence row with:

    python -m experiments.task5_forecasting_baselines --models persistence \
        --decimate 10 --append

`--append` merges into the existing results JSON and refuses to run if the
window geometry differs from the stored configuration, so the appended row is
guaranteed to have been scored on the same windows as the six original ones.

---

## Things a reviewer will ask that are not yet answered

- **Single machine.** Everything here is pelletizer-I. Phase IV.
- **Single seed** on Task 5 and a single block-sampling seed on Task 4. No
  confidence intervals yet. Phase V; the `--seeds` flag exists for it.
- **k=4 is assumed, not selected.** Task 4 fixes k=4 for all methods to keep
  the comparison fair, but the WORKING/PEAK_LOAD split is doing no useful work
  and is the source of most pooled flicker. A k-selection experiment (is k=3
  better for this machine?) belongs with the ablation in Phase IX.
- **The minimum-dwell constant is a hyperparameter.** It is set to the flicker
  threshold to avoid tuning it to the metric, but 10 s is still a choice.
  A hidden semi-Markov model would learn the duration distribution instead.
