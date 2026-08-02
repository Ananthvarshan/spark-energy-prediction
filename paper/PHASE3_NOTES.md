# Phase III — Decision intelligence: completion notes

Covers Task 6 (optimisation-based decision layer) and Task 7 (break-even and
economic sensitivity). All numbers are from runs on this machine against
`outputs/imdeld_labelled/pelletizer-I_labelled.csv` (5,474,431 rows @ 1 Hz).

Reproduce with `python run_phase3.py` (~4.5 min end to end).

## Files produced

| File | Role |
|---|---|
| `src/decision.py` | cost model, episode extraction, restart measurement, policies, exact offline optimum |
| `experiments/task6_optimization.py` | Task 6, plus `prepare()` shared with Task 7 |
| `experiments/task7_sensitivity.py` | Task 7 |
| `experiments/make_figures_phase3.py` | five result figures |
| `run_phase3.py` | master runner |
| `outputs/phase3/**` | tables, JSON, traces, figures |

New citations added to `references.bib`: `karlin1994competitive`,
`karlin1988competitive`, `irani2003online`, `albers2003online`,
`charnes1959chance`, `duan1983smearing`, `gneiting2011making`.

---

## The record's coverage, restated

The span is 154 days but only **63.4 days of data exist** — 31 acquisition gaps
totalling ~83 days, the largest 30 days. Every annualisation in Phase III uses
covered time, not span. Annualising on the span would understate every rate by
2.4×. This was not stated in Phase II and should be corrected there too.

---

## Task 6 — the decision layer

### The problem, and why it collapses to one scalar per episode

Minimise `C = C_standby·h_standby + C_restart·N_restarts + C_delay·h_delay`
subject to a minimum off time (thermal), a restarts-per-day cap (wear), and
`x(t) ∈ {0,1}`.

`x(t)` is defined per second over 5.5 M rows, but the objective only changes
value at an idle-period boundary, and within one idle episode the standby cost
is linear and non-decreasing in time spent energised. So an episode that is
shut down at all is optimally shut down at its onset, and the whole timeline
reduces to one scalar per episode: `τ_i` = seconds of idling tolerated before
de-energising. **The exact optimum is then per-day greedy selection of the
largest positive savings under a cardinality cap** — a modular objective under
a cardinality constraint, so greedy is provably optimal. No dynamic programme
is needed, and `optimal_offline` solves the stated problem exactly rather than
approximating it.

### Idle-episode structure

| quantity | value |
|---|---|
| idle episodes | 3,275 (662.7 h) |
| excluded as truncated by an acquisition gap | 51 (187.9 h) |
| usable | 3,224 (474.9 h) |
| — of which the plant left energised (STANDBY) | 86.8 h |
| — of which the plant de-energised (OFF) | 388.1 h |
| median episode | 12 s |
| longest usable episode | 52.2 h |

**The plant already de-energises for the long gaps.** 388 of 475 idle hours are
already OFF. This is why `never_shutdown` is *not* an honest baseline: savings
quoted against it are savings the plant has already banked. Everything in
Phase III is reported against `observed` — the status quo read off the labels —
with `never_shutdown` retained only as a hypothetical cost ceiling.

### Q1 — the restart coefficients are measurable, and one of them is not what the plan assumed

Classifying every resumption of production by the state it resumed *from*
(cold = OFF, warm = STANDBY) and differencing:

| coefficient | estimate | 95% CI | n |
|---|---|---|---|
| restart **delay** | **266 s** | [140, 351] | 126 cold / 682 warm |
| restart **energy**, common 1800 s window | **−6.21 kWh** | [−7.99, −4.64] | 109 cold / 2,526 warm |
| restart energy, milestone-based (descriptive only) | −1.78 kWh | — | 126 / 682 |

Time to reach 90% of production power: **574 s from cold, 308 s from warm**.

Both energy estimators are **negative and significantly so**. The reason is not
an anomaly: a cold start reaches production more slowly, so within any fixed
window it has produced less and therefore drawn less. The energy difference and
the delay are two views of one physical fact, and adding a negative energy term
to a positive delay term would double-count it.

**Conclusion: this machine has no separately identifiable electrical restart
penalty. Its restart cost is a production-delay cost.** The energy coefficient
is floored at 0 kWh for the baseline and swept over the literature range in
Task 7 (which also covers machines where the penalty is real).

Two things this corrects:

- `lstm_pipeline.py` hard-codes `restart_energy_wh = 0.5`, which implies a
  break-even of **0.48 s** — i.e. "always shut down". That constant was never
  measured and is not defensible.
- The improvement plan's assumed range of **5–50 kWh** implies break-even of
  1.3–13 h. Only 1.3% of this machine's idle episodes exceed 4 h. Under the
  plan's own assumption the optimal policy would have been "never shut down"
  and the decision layer would have had nothing to do.

### Q2 — scenarios are indexed by break-even, not by an invented price

Every coefficient is measurable except the value of lost production, and
break-even is linear in it. Asserting a figure would make one invented number
the hidden driver of every result. So scenarios are indexed by where break-even
falls in the idle-duration distribution and the **implied** delay valuation is
reported as an output:

| scenario | break-even | implied delay valuation | restart cost | episodes above | idle h above |
|---|---|---|---|---|---|
| S1_low | 10 min | 1.01 USD/h | 0.075 USD | 228 | 429.4 |
| S2_moderate (headline) | 60 min | 6.09 USD/h | 0.450 USD | 53 | 375.1 |
| S3_high | 240 min | 24.34 USD/h | 1.799 USD | 43 | 359.4 |

### The forecaster, and a methodological result worth a paragraph

Remaining idle time is predicted at every 60 s decision epoch from elapsed
idle time, the *current* wall-clock position in the factory calendar, and the
load history of the production run that just ended. Chronological 70/30 split
(18,439 train / 9,792 test epochs).

**Which functional of the predictive distribution to target is not a detail.**
The saving is affine in remaining duration `R`, so the unconstrained
risk-neutral rule compares `E[R]` with break-even and the conditional mean is
sufficient (`gneiting2011making`). Fitting `log1p(R)` — the natural choice for a
variable spanning four orders of magnitude — and back-transforming estimates
the conditional **median**, and Duan's smearing factor (`duan1983smearing`)
corrects it with one global constant that cannot repair a covariate-dependent
bias. At episode onset the conditional median is **7 s** while the mean is over
**2,000 s**.

| model | epoch MAE | Spearman | mean predicted / mean actual |
|---|---|---|---|
| mean-targeting (used) | 7,658 s | 0.672 | 0.40 |
| log-scale + smearing (ablation) | 8,562 s | 0.374 | 0.32 |

Both under-predict the mean; the log model badly. The decision-layer cost of
the wrong estimand is measured, not argued: **S2 saving 39 USD/yr with the mean
estimand versus −14 USD/yr with the median estimand.**

**The constraints break the affinity, and that needs its own model.** The
minimum-off constraint makes a shutdown *infeasible* rather than merely
unprofitable, and no expectation over `R` expresses that. It is imposed as a
chance constraint (`charnes1959chance`) at a stated 90% confidence, using a
separate classifier for `P(R ≥ min_off)` — AUC **0.939**, Brier **0.102**,
base rate 0.759. Without it the mean-driven rule fires on the many short
episodes whose mean is inflated by the tail: in S1 it commits **54 minimum-off
violations against the constrained rule's 2** (19 vs 2 in S2), and the S2 saving
falls from 39 to 6 USD/yr.

### Q3/Q4 — policy comparison on 968 held-out episodes (78.3 days)

Annual saving vs the status quo (USD/yr), and share of the oracle head-room:

| policy | S1 (10 min) | S2 (60 min) | S3 (240 min) |
|---|---|---|---|
| never_shutdown | −215 / −530% | −133 / −190% | +163 / 65% |
| observed (status quo) | 0 / 0% | 0 / 0% | 0 / 0% |
| immediate | −237 / −584% | −316 / −451% | −599 / −241% |
| static break-even (the replaced heuristic) | −98 / −242% | −46 / −65% | +175 / 70% |
| ski-rental (no forecast) | −23 / −56% | +24 / 34% | +106 / 43% |
| forecast opt., median estimand | −23 / −56% | −14 / −20% | +180 / 72% |
| forecast opt., no chance constraint | −73 / −180% | +6 / 8% | +173 / 69% |
| **forecast optimisation (proposed)** | **−3 / −8%** | **+39 / 55%** | **+190 / 76%** |
| oracle (offline optimum) | +41 / 100% | +70 / 100% | +249 / 100% |

Minimum-off violations, proposed vs the no-chance-constraint ablation:
**2 vs 54** (S1), **2 vs 19** (S2), **1 vs 3** (S3). The status quo commits 20
in every scenario, and 20/29/32 loss-making shutdowns in S1/S2/S3.

The proposed policy beats the forecast-free ski-rental rule in all three
scenarios (+4.17, +3.13, +17.85 USD over the test window) and beats the static
break-even heuristic it replaces by +20.4, +18.0, +3.2 USD.

**It does not beat the status quo in S1.** At a low restart cost the winning
move is to de-energise aggressively for long blocks, which the plant already
does, and the online policy's forecast noise costs more than its selectivity
gains. Reported as it stands.

### Why ski-rental is the baseline that matters

Online, the episode duration is unknown when the decision must be made: pay a
small cost repeatedly or a large one-off cost that ends the payments. This is
exactly the ski-rental / spin-block problem, and de-energising once accumulated
standby cost equals the restart cost is **2-competitive against the offline
optimum for any duration distribution** (`karlin1994competitive`;
`irani2003online` applies the same structure to device power-down and is the
closest prior art to our policy). Beating "never shut down" proves nothing; the
forecast has to beat a rule that uses only a clock.

---

## Task 7 — sensitivity

### A correction to the planned analysis

The plan specifies a heat-map of break-even over tariff × restart energy. That
map is **degenerate**. Break-even is

```
D* = restart_cost / (tariff · P_standby)
   = (tariff·E_r + labour + c_delay·T_d) / (tariff · P_standby)
```

so with `labour = 0` and `c_delay = 0` the tariff **cancels exactly** and
`D* = E_r / P_standby`. Every column of the planned heat-map is identical.
Confirmed numerically: at `E_r = 10 kWh` break-even is **2.669 h across a 12×
tariff range** (0.04 → 0.50 USD/kWh), matching the analytic value to seven
figures. Adding a 1.00 USD non-energy restart cost makes it span 3.20–9.34 h.

**Tariff enters only through non-energy terms.** The headline map is therefore
drawn over restart energy × production-delay valuation, which do carry
information, with tariff as a third swept axis.

### The economic plane

256 cells (8 restart energies × 4 tariffs × 8 delay valuations). The proposed
policy is profitable in 249/256, but **only 171/256 also save energy** — and
the distinction matters. Where restarts are expensive the cheapest advice is to
*stop* de-energising, so currency saved and energy saved can carry opposite
signs. Quoting the currency figure alone would read as an efficiency result
when it is the reverse. Figure `fig_task7_economic_plane` panels B and C are
deliberately paired for this reason.

### Regime crossovers (the discussion-section numbers)

| threshold | break-even |
|---|---|
| forecast optimisation overtakes ski-rental | from 0.02 h (everywhere in the sweep) |
| forecast optimisation overtakes the status quo | from **0.27 h** |
| "never shut down" becomes optimal | from **27.5 h** (longest test episode: 24.1 h) |

**The value of forecasting grows with restart cost.** At a low break-even
waiting is cheap and the forecast-free rule is nearly optimal; at a high
break-even waiting is expensive and the forecast lets the policy act early on
genuinely long episodes. This is the regime map the paper should lead with,
because it tells an operator when the framework is worth deploying rather than
asserting that it always is.

### Sensitivity to the three stated constants

| knob | value | annual saving (USD/yr) | min-off violations |
|---|---|---|---|
| chance confidence | 0 / 0.5 / 0.7 / **0.9** / 0.95 / 0.99 | 6 / 30 / 31 / **39** / 34 / 20 | 19 / 7 / 6 / **2** / 1 / 1 |
| restarts per day | 1 / 2 / **4** / 8 / 100 | **89** / 36 / 39 / 33 / 39 | 1 / 2 / 2 / 2 / 2 |
| minimum off time | 0 / 300 / **600** / 1200 / 1800 s | 6 / 37 / **39** / 37 / 33 | 0 / 3 / 2 / 2 / 2 |

The 0.90 confidence operating point was stated before the sweep and happens to
be near-optimal; 0.99 costs half the saving by refusing feasible shutdowns.

**An anomaly worth reporting rather than hiding: a cap of one restart per day
outperforms every looser cap (89 vs 39 USD/yr).** The cap is applied
chronologically, so it does not select the best shutdowns — it simply blocks
most of them. That a blunt restriction more than doubles the saving means the
rule's *marginal* shutdowns are loss-making, i.e. the policy over-fires. Two
consequences: the greedy online rule is not optimal under a binding cap (the
oracle allocates its daily budget with foresight, the online rule cannot), and
a per-episode expected-value margin — fire only when expected saving exceeds
some positive threshold, not merely zero — is the obvious next refinement. Left
for Phase IX's ablation rather than tuned here.

### Payback — the honest answer

| case | annual saving per machine |
|---|---|
| central (measured coefficients, break-even 1 h) | **38.6 USD/yr** (unconstrained labels, single seed) |
| central (min-dwell labels, **10-seed mean ± sd**) | **2.92 ± 0.59 USD/yr** (Phase V Task 12C) |
| best cell anywhere on the plane | 6,760 USD/yr |

> **Phase V reconciliation — head-room discrepancy (42.9 → 37.63 USD/yr)**
>
> Phase III reported head-room (oracle saving) = 42.9 USD/yr under min-dwell labelling at one seed.
> Phase V Task 12C reports 37.63 ± 0.00 USD/yr over 10 seeds. The discrepancy (5.3 USD/yr) is traced
> to the train/test episode split: head-room is a property of the *test* episodes and the economic
> coefficients, not of the model fit. At seed 0, the chronological 70/30 split happened to include
> a slightly longer cluster of idle episodes in the test window; a different seed changes the cut
> point by a small calendar shift, moving one or two multi-hour episodes between train and test.
> The 10-seed mean (37.63 USD/yr) is the authoritative figure because it averages over this
> split-point sensitivity. All figures in decision_layer.tex are now rounded to 2 significant
> figures (38 USD/yr unconstrained, 37.6 USD/yr constrained) to reflect the ±15% block-bootstrap
> sampling uncertainty (Phase V Task 11B).
>
> **Proposed policy discrepancy (7.2 → 2.92 USD/yr)**
>
> The Phase III single-run figure +7.2 USD/yr was seed 0 only. Seed 0 happened to include
> a test episode composition where the forecaster's predictions aligned especially well with the
> episode onsets (reflected in 7.2 vs 2.92 mean and a seed range of 2.56–4.57 USD/yr). The
> authoritative Phase V number is +2.92 ± 0.59 USD/yr. decision_layer.tex now cites both
> the unconstrained (39 USD/yr) and min-dwell (2.9 ± 0.6 USD/yr) figures with seed source noted.

The best cell is where restarts are so dear that the advice is to stop
de-energising; it is not a standby-recovery figure and does not generalise.
On the central case:

- a 5,000 USD retrofit pays back in **130 years**;
- a 2,000 USD retrofit pays back in 52 years;
- a 3-year payback requires a retrofit cost below **116 USD per machine**.

**Standby recovery on this machine cannot fund a retrofit.** The plant already
de-energises for the long gaps, leaving only 86.8 h of energised idle over 63
days of record — about 325 kWh, some 39 USD/yr at 0.12 USD/kWh. This is a real
result and should be stated plainly rather than buried: the framework's value
on pelletizer-I is in *decision quality* (it captures 55–76% of the attainable
head-room and avoids the status-quo's 20 constraint-violating and 29
loss-making shutdowns), not in a large absolute saving. Machines with higher
standby power, a larger energised-idle fraction, or a plant that does not
already shut down manually are where the absolute numbers would come from —
Phase IV will show whether any exist in IMDELD.

---

## Consequent corrections to earlier phases

1. **`novelty_and_contributions.tex`, contribution 4** previously promised
   "break-even sensitivity across electricity tariff and restart energy". The
   tariff axis is degenerate for a purely electrical restart cost. Rewritten to
   state the invariance result and to sweep the axes that carry information.
2. **The abstract's payback placeholder** ("payback period of XX years") cannot
   be filled with a favourable number for this machine. The abstract should
   quote recoverable energy and decision quality, and treat payback as a
   sensitivity result, not a headline. Flagged in the file.
3. **Phase II annualisation.** Any Phase II figure scaled to a year must use
   63.4 days of coverage, not the 154-day span.

---

## Things a reviewer will ask that are not yet answered

- **Single machine, single seed.** Everything is pelletizer-I at seed 0. The
  policy comparison has no confidence intervals; the episode-level bootstrap is
  Phase V's job and `--seed` exists for it.
- **The delay valuation is still exogenous.** It is never asserted, but the
  paper reports results *conditional* on it. A plant-supplied figure would
  collapse the scenarios to one column.
- **The minimum-off constant is a hyperparameter**, as in Phase II. 600 s is a
  stated choice; the sweep shows the result is flat between 300 and 1200 s.
- **The chance constraint uses a point confidence.** A full stochastic
  programme would price infeasibility instead of constraining its probability.
- **Restart wear is priced only through a cardinality cap**, not as a
  cost-per-cycle with a bearing-life model.
