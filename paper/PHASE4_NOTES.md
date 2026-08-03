# Phase IV — Experimental validation: completion notes

Covers Task 8 (multi-machine analysis), Task 9 (cross-machine
generalisation) and Task 10 (cross-dataset validation). Every number below is
from runs on this machine.

Reproduce with `python run_phase4.py` (~4 h end to end; `--skip task9b` cuts
it to about 1.5 h).

## Files produced

| File | Role |
|---|---|
| `src/labelling.py` | the uniform labelling procedure applied to every record |
| `tests/test_labelling.py` | equivalence tests against `validate_gmm.py` and Phase II |
| `experiments/task8_multi_machine.py` | Task 8 |
| `experiments/task9_cross_machine.py` | Task 9, parts A and B |
| `experiments/task10_cross_dataset.py` | Task 10 |
| `experiments/make_figures_phase4.py` | six result figures |
| `run_phase4.py` | master runner |
| `outputs/phase4/labelled/*.parquet` | the ten labelled records (680 MB) |
| `outputs/phase4/task8/**` | per-machine characterisation, BIC scans, decision runs, three tables |
| `outputs/phase4/task9/part_a/**` | decision-forecaster transfer matrix |
| `outputs/phase4/task9/part_b/**` | eight trained models, cached test windows, transfer matrix |
| `outputs/phase4/task10/**` | cross-dataset characterisation and transfer |
| `outputs/phase4/figures/**`, `logs/**` | figures and run logs |

`experiments/common.py` gained the eight-machine registry, the two SPARK
datasets, a parquet-aware `load_labelled(source=...)` and `degenerate_reason`.
`experiments/task6_optimization.prepare` gained `source` and `split_ts`
arguments; both default to Phase III's behaviour.

---

## The labelling procedure, and why Phase IV re-labels pelletizer-I too

The Phase I–III labels exist only for pelletizer-I and were produced by a
script whose configuration has since moved (`validate_gmm.K_RANGE` is now
`[3]`; the export in `outputs/imdeld_labelled/` is a four-state one). Comparing
that export against seven fresh labellings would confound machine differences
with procedure differences, so `src/labelling.py` re-labels all eight machines
identically:

```
load → spike mask (rolling MAD) → OFF denoise → log1p + standardise
     → GMM emissions → GaussianHMM, transitions only → Viterbi per segment
     → minimum-dwell constraint (10 s)
```

Preprocessing reproduces `validate_gmm.py` exactly; `tests/test_labelling.py`
checks that claim rather than asserting it (6/6 pass, including a 200-case
equivalence test of the minimum-dwell operator against Phase II's version).
The decoding follows Phase II's own conclusions: emissions warm-started from a
GMM and held fixed (Task 4 correction 2), and an explicit minimum dwell
(Task 4c). Labelling all eight machines takes 22 min and produces 680 MB of
parquet rather than 3.8 GB of CSV.

**Three departures from `validate_gmm.py`, each verified rather than assumed:**

1. *Blank currents.* The milling-machine meters leave `current` empty on
   **36.1% and 36.3%** of rows — matching their OFF shares of 36.1% and 36.3%
   almost exactly, i.e. the meter blanks the current whenever the machine is
   de-energised. On every one of those rows apparent power is exactly 0 VA
   with voltage present, so `S = V·I` admits only `I = 0 A`.
   `validate_gmm.py` leaves those cells NaN, through which a GMM cannot be
   fitted at all — which is presumably why no milling machine was ever run
   through it. They are filled with zero and `tests/test_labelling.py`
   asserts that every filled cell met that condition. Dropping them instead
   would have deleted a third of each record precisely where the machine is
   off, biasing every state-time total.
2. *Timestamps.* IMDELD spans a daylight-saving change, so the column carries
   mixed UTC offsets and `pd.to_datetime` falls back to a per-element Python
   parse — 115 s per million rows. Splitting the naive part from the offset is
   exact and ten times faster; verified element-for-element.
3. *float32 storage.* Eight float64 records do not fit in this machine's
   memory. The one measurable consequence is that ~2 rows per 10⁵ fall on the
   other side of the `5×MAD` spike threshold.

**Continuity check on pelletizer-I** (the only machine both labellings cover):
row agreement **98.88%**, STANDBY-vs-rest agreement **99.84%**, ARI **0.978**,
STANDBY **88.8 h** against the export's **87.2 h**. The uniform labeller
reproduces the pipeline's own export, so a Phase IV number that differs from a
Phase III number is a consequence of the analysis, not of the labels — except
where §"the min-dwell constraint moves the decision layer" says otherwise.

---

## Task 8 — the eight machines

### The state model holds on the four driven machines and fails on the four auxiliaries

| machine | covered d | STANDBY h | h/covered day | P_standby | physics | PF sep. | I ratio | flicker | Otsu PF | M1 agree |
|---|---|---|---|---|---|---|---|---|---|---|
| pelletizer-I | 63.4 | 88.8 | 1.40 | 4 208 W | **5/5** | 0.722 | 0.359 | 0.00% | 0.474 | 99.4% |
| pelletizer-II | 62.4 | 115.6 | 1.85 | 4 539 W | **5/5** | 0.714 | 0.363 | 0.00% | 0.462 | 99.7% |
| milling-I | 12.2 | 67.1 | 5.52 | 4 553 W | **5/5** | 0.609 | 0.426 | 0.17% | 0.477 | 98.1% |
| milling-II | 12.2 | 64.0 | 5.27 | 5 447 W | **5/5** | 0.523 | 0.326 | 0.00% | 0.583 | 97.7% |
| exhaust-fan-I | 63.7 | 555.6 | 8.73 | 0.3 W | 4/5 | 0.912 | **0.008** | 0.00% | 0.705 | 96.6% |
| exhaust-fan-II | 62.3 | 90.7 | 1.45 | 0.2 W | 3/5 | 0.624 | **0.002** | 0.00% | 0.632 | **55.1%** |
| dpc-I | 63.7 | 57.2 | 0.90 | 0.3 W | 3/5 | 0.307 | **0.009** | 0.00% | 0.236 | 90.3% |
| dpc-II | 62.5 | 72.2 | 1.16 | 9.3 W | 3/5 | **−0.052** | **0.007** | 0.00% | 0.232 | 85.4% |

**The two pelletizers and the two milling machines pass all five physics
checks.** Their STANDBY clusters sit at 4.2–5.4 kW with a power factor of
0.11–0.17 against 0.64–0.65 when loaded, and a no-load current ratio inside the
motor band [0.25, 0.50]. That is the textbook signature of an energised
induction motor drawing magnetising current, found independently on four
machines, and it is the strongest evidence in the paper that the state
definition is physical rather than fitted.

**On the fans and contactors there is no energised-idle state to find.** Their
"STANDBY" clusters sit at 0.2–9.3 W with 0.09–0.14 A: the model is splitting
the OFF band in two because it was told to find four states. The checks catch
it — the current ratio is 0.002–0.009 against an acceptance band of
[0.25, 0.50] on all four, and dpc-II additionally has a *negative* power-factor
separation. **This is the physics battery doing the job it was built for**, and
it is worth stating plainly: the framework's state definition is specific to
motor-driven loads, and it says so on the machines where it does not apply
rather than returning four confident states regardless.

Two details worth a sentence each in the paper. **Exhaust-fan-I's state names
are mis-assigned, and by a mechanism the paper should describe.** Its "OFF"
cluster (0.8% of the record) carries 6.6 A at 0 W and PF 0 — standing reactive
current with no real power — while its "STANDBY" cluster (36.4%) sits at 0.3 W
and 0.13 A and is the machine's true de-energised state. The canonical names
are handed out by mean active power, so a near-zero-power cluster carrying
reactive current ranks below the genuine off state and takes its name. This
does not affect any hour total (both are non-productive), but it is why the
current-ratio check fails in the direction it does, and it is a real limitation
of ranking states by active power alone.
Exhaust-fan-II is the one machine where the independent Otsu power-factor
classifier agrees with the state model on only **55%** of energised rows (the
other seven: 85–99.7%), i.e. the two disagree exactly where the model has
nothing physical to lock onto.

### The minimum-dwell result generalises

Flicker *before* the constraint, measured on every row of every record:
**26.7% (milling-II) to 77.7% (exhaust-fan-II)**, median 50.6%. After it:
**0.00% on seven machines and 0.17% on milling-I**, at a cost of relabelling
**0.15–6.1%** of rows. Phase II established on one machine that the transition
prior cannot control flicker and an explicit dwell constraint can; that now
holds on eight records with duty cycles that differ by an order of magnitude.

### BIC does not select a state count here

The labeller's own scan (k ∈ 2..5) returned k = 5 on every machine — the
boundary. Widening it to k ∈ 2..8 gives **k = 8 on seven machines and k = 7 on
pelletizer-II**, with BIC still falling at the boundary in seven of eight
cases; the drop from k = 4 to k = 8 is 3.1–33.1% of the criterion's own value.

This is a result about the criterion, not the data. At n > 10⁶ the BIC penalty
is negligible against the likelihood gained by splitting a non-Gaussian
component, so a mixture model keeps buying components to approximate a
distribution that is not a mixture of four Gaussians. **The physics vetoes that
`validate_gmm.py` applies on top of BIC are therefore not a belt-and-braces
addition: on these records they are the thing actually doing the selecting.**
Phase IV fixes k = 4 everywhere for comparability, and this is the honest
statement of what that costs.

### Which machines are hard, and why

Difficulty is measured on three axes: the separability of the no-load/load
boundary (Cohen's d and Bhattacharyya distance on log power), record quality
(coverage fraction, spike rate, segment count), and agreement with the
independent Otsu classifier.

The ordering is **not** the one the improvement plan predicted. The plan
expected the pelletizers to be medium and the fans and conveyors hard because
of low power. What the data show is that the fans and contactors have the
*largest* separability numbers of all (Cohen's d of 9.3–75.0 against the
pelletizers' 4.3–7.2) while failing the physics checks — because the
separation being measured is OFF against running, not no-load against loaded.
**Separability and validity are close to orthogonal here**, which is the
argument for keeping a physics battery rather than an internal cluster-quality
score: silhouette, BIC and Cohen's d are all maximised by the partitions that
are physically wrong.

The two milling machines are hard for a different reason: their records cover
12.2 of 43 days in **1 082 and 580 fragments**, against 66–74 fragments for the
five-month records. That fragmentation is what breaks the decision layer on
them (below), not the state model, which passes 5/5 on both.

### The decision layer exists on three of eight machines

| machine | idle h | of which energised | P_standby | restart delay | n_cold | decision problem |
|---|---|---|---|---|---|---|
| pelletizer-I | 474.1 | 87.8 (18.5%) | 4 208 W | 270 s | 116 | **well posed** |
| pelletizer-II | 459.7 | 111.1 (24.2%) | 4 539 W | — | **0** | degenerate |
| milling-I | 136.6 | 66.7 (48.8%) | 4 553 W | 8 s | **3** | well posed, unidentified |
| milling-II | 134.0 | 63.8 (47.6%) | 5 447 W | 11.5 s | **18** | well posed, unidentified |
| exhaust-fan-I | 382.6 | 370.7 (96.9%) | 0.3 W | −838 s | 118 | degenerate |
| exhaust-fan-II | 363.8 | 65.7 (18.1%) | 0.2 W | — | 259 | degenerate |
| dpc-I | 465.5 | 54.9 (11.8%) | 0.3 W | 165 s | 89 | degenerate |
| dpc-II | 480.1 | 66.3 (13.8%) | 9.3 W | −166 s | 95 | degenerate |

Five machines are marked degenerate by an explicit test
(`experiments.common.degenerate_reason`) rather than by inspection: four
because standby power is below 1 W, so `standby_cost_per_s ≈ 0`, the break-even
point is infinite and every currency figure would be an artefact of dividing by
numerical zero; and pelletizer-II because it is **never once observed resuming
production from OFF** in five months, so the cold-versus-warm restart contrast
cannot be measured at all. Two machines return a *negative* restart delay
(cold starts reaching production faster than warm ones), which is not a
plausible physical reading and is a further symptom of the same problem.

**Phase III's open question is now answered, and the answer is no.** It ended
by noting that pelletizer-I's 86.8 h of energised idle cannot fund a retrofit
and that "machines with higher standby power, a larger energised-idle fraction,
or a plant that does not already shut down manually are where the absolute
numbers would come from — Phase IV will show whether any exist in IMDELD."
None do. The three machines with a well-posed problem have head-rooms of
**37.6, 24.9 and 25.2 USD/yr** (Phase III's 70/30 split on episode count, held
identical across machines so the column is comparable; pelletizer-I's figure
becomes 42.9 under the aligned-instant split used in the labelling comparison
below, which does not change the ordering); the facility's total attainable saving from
standby recovery, summed over every machine where the question is even
meaningful, is under **90 USD/yr**. The framework's contribution on this
facility is decision quality, not recovered energy, and the paper should say so
in the abstract rather than at the end of the discussion.

The milling machines are worse than that: the proposed policy *loses*
1 041 and 1 256 USD/yr there against a head-room of 25 USD/yr. The cause is
visible in the same table — a forecaster with MAE 18 200 s and Spearman 0.34–0.43
(against 9 034 s and 0.58 on pelletizer-I), fitted on a record broken into
1 082 fragments, over-predicting remaining idle time and firing shutdowns that
the realised episode does not support. **A negative result, reported as it
stands**: the decision layer inherits its forecaster's quality, and on a record
this fragmented the forecaster is not good enough to act on.

### The min-dwell constraint moves the decision layer, not just the labels

Same machine, same decision layer, two labellings. Both runs are cut at the
**same instant** (2018-01-16 22:24, the timestamp 70% of the way through the
record's covered rows) rather than at the 70th percentile of the episode index:
the two labellings do not produce the same number of episodes, so a count-based
split would test them on different calendar windows and confound the labelling
with the weather of one month.

| quantity | Phase I–III export | Phase IV labelling |
|---|---|---|
| idle hours | 474.9 | 474.1 |
| energised idle (STANDBY) h | 86.8 | 87.8 |
| standby power | 3 747 W | 4 208 W |
| restart delay | 266 s | 270 s |
| **usable idle episodes** | **3 224** | **1 312** |
| **median episode** | **12 s** | **90 s** |
| forecast MAE | 7 922 s | 8 098 s |
| head-room (S2) | 71.4 USD/yr | **42.9 USD/yr** |
| proposed policy (S2) | +40.9 USD/yr (**57.3%** of oracle) | +7.2 USD/yr (**16.8%**) |
| ski-rental (S2) | +28.7 USD/yr | −0.3 USD/yr |

(The count-based split, which is what Phase III ran, gives 70.1 / +38.6 / 55.0%
against 37.6 / +2.9 / 7.6% — the same conclusion, and the +38.6 USD/yr at 55%
of oracle reproduces Phase III's published figure exactly, which is the check
that the two runs differ only in their labels.)

The physical totals are stable to within 2%: the two labellings agree about how
much idle time exists and how much of it the plant leaves energised. What they
do not agree about is how that time is **partitioned into episodes**, and the
episode is the unit the decision layer bids on. Removing sub-10-second state
runs merges 3 224 episodes into 1 312 and raises the median from 12 s to 90 s.

**Both of Phase III's headline decision numbers are sensitive to this.** They
should be reported against the labelling they were computed on, and the paper
should carry both columns rather than the more favourable one. The
interpretation is not that Phase III was wrong — it is that a decision layer
evaluated on flickering labels is partly bidding on episodes that the flicker
created, and the honest version of the result is the one computed on labels
whose dwell times are physically admissible.

---

## Task 9 — cross-machine generalisation

### Part A: the decision forecaster

Every pair of the eight machines, in both directions, with the target
machine's own economics and two input variants (raw, and with the one
absolute-power feature rescaled by the target's own median productive power —
a constant available from unlabelled history). Sixty-four cells × 2.

**Transfer costs almost nothing, because there is almost nothing to transfer.**
Median MAE: within-machine **11 118 s**, same-type **11 471 s**, cross-type
**11 653 s** — a 4.8% degradation from the hardest possible transfer. On two
targets a *foreign* model is the best one: pelletizer-I's own forecaster scores
MAE 9 034 s while dpc-I's scores 8 695 s on it. A model that transfers this
freely between a 40 kW pelletizer and a contactor is not carrying
machine-specific structure; it is carrying the plant calendar, which they
share.

**The forecaster's own feature importances say so directly.** In Phase III's
fit the four calendar features carry **94.5%** of the gain (`onset_dow` 0.473,
`cur_is_weekend` 0.291, `onset_hour` 0.106, `cur_is_open` 0.075) and the
machine's own electrical history carries **2.4%** (`prev_productive_power_w`).

> **CORRECTED in Phase VII (PHASE6_NOTES correction 1). Do not quote 94.5%.**
> That figure is split gain, and Phase VI showed gain is misranking this
> model: `cur_is_weekend` takes 63.2% of the gain and 5.1% of the Shapley
> attribution, because it is a binary split near the root on a partition
> later splits refine. The figure the paper quotes is the Shapley
> attribution: **82.4% calendar (96.7% of gain) under the minimum-dwell
> labelling**, and **78.0% (96.2%)** on the Phase I–III labelling this
> sentence was originally measured on. The machine's own production history
> carries **13.3%**, not 2.4%. The *conclusion* below is unchanged and is now
> better supported: a model that is four-fifths plant calendar transfers
> within the facility and nowhere else. `paper/crossmachine.tex` §
> "What the forecaster is actually using" carries the corrected statement.

A model that is four fifths calendar will transfer between any two machines on
the same shift pattern and will transfer to nothing else — which is exactly the
pattern Part A measures here and Task 10 measures across sites. Note the
contrast with Task 3, where temporal features were nearly irrelevant (1–4%) to
the *state* split: the two layers of the framework depend on almost disjoint
information, and only the state layer is physics-driven.

**The scale correction does not help, which refutes the obvious hypothesis.**
Rescaling changes median cross-type MAE from 11 653 s to 11 597 s (0.5%) and
median cross-type Spearman from 0.357 to 0.287. The transfer loss is not a
units mismatch.

**In currency, on the one target where the decision problem is well posed and
the head-room is not noise** (pelletizer-I, head-room 37.6 USD/yr):

| source | annual saving | % of oracle | beats ski-rental |
|---|---|---|---|
| pelletizer-II (same type) | **+8.15** | **21.7%** | yes |
| pelletizer-I (itself) | +2.86 | 7.6% | yes |
| exhaust-fan-I | −1.38 | −3.7% | no |
| dpc-II | −2.51 | −6.7% | no |
| dpc-I | −3.34 | −8.9% | no |
| exhaust-fan-II | −20.4 | −54% | no |
| milling-II | −38.1 | −101% | no |
| milling-I | −45.6 | −121% | no |

The peer machine's forecaster **beats the machine's own** (+8.15 against
+2.86). With 1 312 episodes and a 30% test split there is no basis for calling
that a real ordering — the honest reading is that same-type transfer costs
nothing measurable, and that both are close to the forecast-free floor.
Cross-type transfer is another matter: five of six cross-type sources lose
money, and only **5.6% of cross-type pairs beat ski-rental** against 33% of
within-machine pairs.

**The deployable conclusion:** a forecaster may be moved between machines of
the same type without refitting, must not be moved between types, and on this
plant is worth little either way against a rule that uses only a clock.

### Part B: the sequence forecaster

The proposed Seq2Seq LSTM trained once per machine under Task 5's protocol
(0.2 Hz after 10× decimation, 600 s lookback, 300 s horizon, 30 epochs with
early stopping) and evaluated on every machine's held-out windows, with two
scaler variants and a **persistence reference** on each target — the horizon
continues the last observed state, which needs no training at all.

Median over the matrix, F1 on the STANDBY class:

| | macro F1 | F1 (STANDBY) | STANDBY-seconds MAE |
|---|---|---|---|
| trained on the target machine | **0.695** | **0.539** | **16.4 s** |
| same-type transfer (source scaler) | 0.511 | 0.479 | 36.7 s |
| cross-type transfer (source scaler) | 0.286 | 0.076 | 70.8 s |
| cross-type transfer (target scaler) | 0.374 | 0.149 | 44.2 s |
| persistence (no training) | 0.629 | 0.463 | 28.8 s |

**This model does not transfer, and the contrast with Part A is the point.**
The decision forecaster moved between machines at a 4.8% cost in MAE; the
sequence model loses 86% of its STANDBY F1 crossing machine types. The two
results are consistent rather than contradictory: what the sequence model
learns *is* machine-specific — the electrical shape of one motor's duty cycle —
while what the decision forecaster learns is the plant calendar, which every
machine in the facility shares. **What transfers here is not machine-specific,
and what is machine-specific does not transfer.**

**Re-standardising to the target helps cross-type transfer and hurts same-type
transfer** (cross-type F1 0.076 → 0.149; same-type 0.479 → 0.305). A source
model applied to a machine of the same type is better off keeping the scale it
was trained on, since the two machines genuinely have similar ratings; applied
across types it is better off being told the new machine's scale. Neither
variant is close to a locally fitted model.

**The training-free baseline is the uncomfortable result.** Persistence reaches
F1 0.463 against the locally trained model's 0.539, and per machine:

| target | own model | persistence | transferred (median) |
|---|---|---|---|
| pelletizer-I | 0.660 | 0.650 | 0.20 |
| pelletizer-II | 0.607 | 0.432 | 0.43 |
| milling-I | 0.515 | 0.504 | 0.20 |
| milling-II | 0.489 | **0.493** | 0.25 |
| exhaust-fan-I | **0.982** | 0.288 | 0.11 |
| exhaust-fan-II | 0.477 | 0.211 | 0.01 |
| dpc-I | 0.402 | 0.012 | 0.08 |
| dpc-II | 0.563 | 0.552 | 0.12 |

On **pelletizer-I — the machine the whole paper is built on — the proposed
Seq2Seq LSTM beats "assume nothing changes for the next five minutes" by 0.010
of F1** (0.660 against 0.650), and on milling-II persistence wins outright.
The model earns its place only on the machines whose STANDBY state is easy for
a different reason (exhaust-fan-I, where STANDBY *is* the off state and 98% F1
is achievable). Only **17.9–21.4% of transferred pairs beat the target's own
persistence baseline**, against 87.5% of locally trained models.

The diagonal reproduces Phase II: Task 5 reported STANDBY F1 = 0.6597 for the
Seq2Seq LSTM on pelletizer-I and this run gives 0.660 on relabelled data, so
the two are measuring the same thing. **What Phase II lacked was this
baseline.** Its six-model comparison ranked neural architectures against each
other and against XGBoost, all of which were trained; none of them was compared
with not training at all. That omission should be repaired in the paper — the
sequence-forecasting contribution as it stands is worth 0.01 F1 on the headline
machine.

---

## Task 10 — cross-dataset validation

Two single-channel active-power records at 5 s for 2024, from a different site
and acquisition system: a CNC machining centre (348 covered days) and a
photovoltaic inverter's AC output (86 covered days). The second is a **negative
control** — a generator, not a driven load, for which no shutdown decision
exists — included because a validation battery is only worth quoting if it can
reject something.

| | CNC (machine) | PV inverter (control) |
|---|---|---|
| channels | 1 | 1 |
| physics checks evaluable | 3/5 | 3/5 |
| checks passed | **3/3** | **2/3** (C5 fails) |
| flicker | 0.00% | 0.00% |
| median dwell | 120 s | 30 s |
| STANDBY | 185.6 h @ 179 W | 41.6 h @ 444 W |
| productive at night (00–05) | 7.9% | **0.0%** |
| productive at midday (10–15) | 34.2% | 54.1% |

**The pipeline runs unchanged on both and produces a plausible-looking
four-state partition for both.** On the CNC that is the right answer: OFF 0 W,
STANDBY 179 W, WORKING 2.24 kW, PEAK 5.70 kW, with a diurnal profile that
looks like a day-shift machine.

**On a solar inverter it is the wrong answer, and the battery nearly fails to
say so.** Two of the five checks — power-factor separation and no-load current
ratio — are undefined without current and power-factor channels, and those are
precisely the two that do not depend on power magnitude, i.e. the two that test
whether STANDBY has been separated from a lightly loaded WORKING state rather
than merely from a lower-power one. Of the three that remain, the control
passes power ordering and OFF-near-zero and fails only variance ordering, by a
2% margin (σ_OFF = 234 W against σ_STANDBY = 229 W). The framework reports
41.6 h of recoverable "standby" on a device that cannot be shut down.

**This is the central Task 10 finding and it is a limitation of the method, not
of the data.** On a single-channel record the state model still produces
states and the decision layer still consumes them, but the physics validation
has lost most of its discriminating power.

Nor does the diurnal profile rescue it. Both records are daytime-active, and
by the midday figure the inverter looks like the *better* machine (54.1%
productive against the CNC's 34.2%); only the night figure separates them at
all, and then by 7.9 points. What actually differs is the SHAPE — the CNC's
profile is a plateau with sharp edges at 04:00 and 15:00 UTC, the inverter's a
smooth bell centred on solar noon — and no test in the battery looks at shape.
A single-channel deployment therefore needs an external admissibility check
(is this a driven load?) that the current framework does not provide, and
Figure `fig_task10_cross_dataset` panel B is the evidence for saying so.

Two further results:

**The decision layer cannot be instantiated on either record.** Both return a
negative cold-versus-warm restart delay, so the same guard that flagged five
IMDELD machines flags both of these. On the CNC the reason is visible in the
episode structure: 229 usable idle episodes with a **median duration of 14.9 h**
against pelletizer-I's 90 s, i.e. a machine that is off far more than it is on
and is rarely observed restarting at all.

**Zero-shot transfer across datasets fails outright.** The pelletizer-I
forecaster on the CNC record scores MAE 120 955 s and Spearman **−0.245**
against the locally fitted 60 386 s and +0.659; rescaling does not help
(−0.234). A *negative* rank correlation means the transferred model orders the
CNC's idle episodes worse than chance would. This is the expected consequence
of Part A's finding: what the forecaster transfers is the IMDELD plant
calendar, and the SPARK site does not keep IMDELD's shifts.

---

## Consequent corrections to earlier phases

1. **Phase III's decision-layer headline is labelling-dependent.** +38.6 USD/yr
   and 55% of oracle were computed on the Phase I–III export; under the
   minimum-dwell labelling that Phase II's own diagnosis recommends, and on an
   identical test window, they are **+7.2 USD/yr and 16.8%** against a
   head-room that falls from 71.4 to 42.9 USD/yr. Both belong in the paper,
   with the episode-count and median-duration rows that explain the
   difference.
2. **Phase III's closing conjecture is refuted.** No IMDELD machine has more
   recoverable standby than pelletizer-I; four have none at all. The facility
   total is under 90 USD/yr, and the paper's economic claim must be framed
   around decision quality and constraint compliance rather than around
   recovered energy.
3. **k-selection.** BIC is still falling at k = 8 on seven of eight machines,
   so the criterion does not select a state count on records of this size and
   the physics veto is load-bearing. Two consequences. In the code,
   `validate_gmm.py`'s test T3 is titled "k Selection (BIC + physical filters
   agree)" — they do not agree, and the filter decides; the title is
   misleading and should say so. In the paper, `related_work.tex` §2.1
   currently reports only that "model order is conventionally selected by the
   Bayesian information criterion or by silhouette analysis", which is true
   but incomplete: one sentence should be added recording that at n > 10⁶ both
   criteria keep buying components on this data, which is what motivates the
   physics veto rather than a purely statistical selection. That is an
   addition, not a contradiction, so the file is left for the next phase that
   touches it.
4. **The five-method proof generalises with one exception.** The Otsu
   power-factor classifier agrees with the state model on 85.4–99.7% of
   energised rows on seven machines and on 55.1% on exhaust-fan-II — the
   machine whose physics checks fail hardest. Method 1's usefulness as an
   independent check is therefore confirmed *and* its failure mode is
   identified.
5. **Task 5's baseline set is incomplete.** It compares six trained
   forecasters and no untrained one. A persistence reference on the same
   windows reaches STANDBY F1 0.650 against the proposed Seq2Seq LSTM's 0.660
   on pelletizer-I, and wins on milling-II. The comparison table should carry
   a persistence row, and the paper should not claim more for the sequence
   model than 0.01 of F1 over doing nothing on the headline machine. This does
   not change the *ranking* Task 5 reported among the trained models.
6. **A merge bug in the continuity check, found and fixed here.** The record
   contains repeated timestamps, so joining two labellings on the timestamp
   alone returns 5.91M rows from two 5.47M-row inputs and inflates every hour
   total computed from it. The join is now on (timestamp, occurrence index).
   Anything elsewhere in the project that merges these records on timestamp
   alone has the same defect.

---

## Things a reviewer will ask that are not yet answered

- **Single seed.** Every Phase IV result is at one seed: one labelling seed
  (42), one decision-layer seed (0), one training seed for the sequence
  models. The block-sampling and episode-level variability is Phase V's job.
- **k = 4 is imposed on records that do not have four states.** The fans and
  contactors would be better described with two, and the BIC scan says the
  driven machines want more than four. A per-machine k, selected under the
  physics veto rather than by BIC, is the obvious Phase IX ablation.
- **The milling records are 28% covered.** Their 5.5 h/day of standby is the
  largest per-day figure among the driven machines, but it rests on 12 days of
  data in 1 082 fragments and should not be annualised.
- **Two datasets is not many.** Task 10's second dataset is single-channel, so
  it can test the pipeline but not the physics battery. A second *six-channel*
  industrial record would test the part of the framework that matters most and
  none was available here.
- **The transfer study is within one facility.** Task 9's eight machines share
  a plant calendar, which is precisely what the forecaster turns out to be
  using. A cross-facility transfer of the decision forecaster is untested; the
  Task 10 result suggests it would fail.
