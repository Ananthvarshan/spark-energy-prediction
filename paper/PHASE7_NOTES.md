# Phase VII — Final paper assembly: completion notes

Covers Task 14A (the outstanding corrections), 14B (the four missing
sections), 14C (assembly), 14D (limitations) and 14E (abstract and
introduction). This phase writes prose and runs checks; it fits no models
and produces no new experimental numbers. Every figure quoted in the new
sections is read from a stored Phase II–VI output, and the ones that
matter were re-read from the source CSV/JSON rather than copied from the
notes — which caught three discrepancies, recorded below.

Reproduce the assembly and its checks with `python generate_paper_draft.py`
(under a second).

## Files produced

| File | Role |
|---|---|
| `paper/abstract.tex` | Task 14E — the abstract |
| `paper/introduction.tex` | Task 14E — §1: context, positioning, contributions, organisation |
| `paper/method.tex` | §3 wrapper: overview, data, the two forecasting problems; `\input`s the labeller |
| `paper/state_layer.tex` | §4: features, state-detection comparison, flicker, ground-truth-free validation |
| `paper/forecasting.tex` | Task 14B — §5 (`sec:forecasting`) |
| `paper/significance.tex` | Task 14B — §7 (`sec:significance`) |
| `paper/multimachine.tex` | Task 14B — §8 (`sec:multimachine`) |
| `paper/crossmachine.tex` | Task 14B — §9 (`sec:crossmachine`) |
| `paper/limitations.tex` | Task 14D — §11 |
| `paper/conclusion.tex` | §12 |
| `paper/main.tex` | Task 14C — **generated**; do not edit by hand |
| `generate_paper_draft.py` | assembly + six static checks |

`paper/novelty_and_contributions.tex` no longer contains body text. Its
abstract and contribution list are superseded by `abstract.tex` and
`introduction.tex`; keeping the same prose in two files guarantees one
copy drifts. What remains in it is the design rationale for the Phase I
reframing and a log of what each later phase changed about it, which is
recorded nowhere else.

Four bibliography entries were added: `holm1979simple`,
`benjamini1995controlling`, `kunsch1989jackknife`, `efron1993bootstrap`.
The significance section used all four procedures and cited none of them.

---

## Task 14A — the corrections, and one more that surfaced

All five outstanding items from `PHASE6_NOTES.md` are applied, not four:
the user's list named 1–4 and correction 5 was still open.

**1. The 94.5% calendar share is gone.** It was split gain, and gain
misranks that model badly enough to be worth saying so explicitly:
`cur_is_weekend` takes 63.2% of the gain and 5.1% of the Shapley
attribution. `crossmachine.tex` § "What the forecaster is actually using"
now states **82.4% of the attribution (96.7% of the gain)** under the
minimum-dwell labelling, 78.0% (96.2%) on the Phase I–III labelling, and
machine history at **13.3%** rather than 2.4%. `PHASE4_NOTES.md` carries a
blockquote at the original sentence marking it corrected, so the note and
the paper cannot drift apart.

**2. The BIC sentence is in `related_work.tex` §2.1.** One paragraph: the
penalty grows as ½p·log n while the likelihood gained by splitting a
non-Gaussian component grows linearly, so at n > 10⁶ the criterion keeps
buying components; BIC is still falling at k = 8 on seven of eight
machines. It closes by naming the accept/reject battery as the
consequence, which is what makes it an argument for the method rather
than an aside.

**3. The multiclass headline is dropped, not updated.**
`state_layer.tex` §4.1 reports the STANDBY-vs-productive target only and
says in one sentence why the four-class ranking is not reported.
`PHASE2_NOTES.md` carries the same statement beside the stored CSV.

**4. The voltage caveat is stated twice, deliberately** — in
`forecasting.tex` (where a reader might otherwise take the model as
portable) and in `limitations.tex` §11.3. 20.0% of the window model's
attribution is supply voltage, which is a property of the site's point of
common coupling and not of the machine, and split gain ranks it tenth of
twenty-five channels and gives no warning.

**5. The minimum-dwell join artefact is now in the method**
(`method_gmm_labeller.tex`), not only in the explainability section. The
operator is described as removing runs; it also, rarely, joins two states
that were not adjacent — three PEAK_LOAD → OFF events in 5.47 M rows.

### A sixth correction, found while writing the method section

**The accept/reject loop was described as a battery of ten tests. It is
five.** `src/labelling.py` calls `experiments.common.physics_compliance`,
which scores C1–C5 (power ordering, OFF near zero, PF separation, no-load
current ratio, variance ordering) at a 0.90 threshold. The T1–T10 harness
in `validate_gmm.py` is a different thing: it is reported per machine and
is not consulted at fit time. The two were conflated in
`method_gmm_labeller.tex`, and the Phase V seed scores prove which is
which — 0.60/0.80/1.00 are multiples of 1/5, not of 1/10.

The section now lists C1–C5 explicitly, names **C3 and C4 as the
load-bearing pair** (the only two independent of power magnitude, hence
the only two that can distinguish energised idle from lightly loaded),
and states the relationship to T1–T10. This matters beyond tidiness: a
reviewer reading the released code would have found the discrepancy, and
the five checks are exactly what Task 8's per-machine "physics 5/5"
column reports.

---

## Task 14B — the four missing sections

Each was written from the stored outputs rather than from the phase
notes, and the source file is named in each `.tex` header.

**`forecasting.tex`** — seven models, mean ± sd over seeds (15 for the
tree, 5 per neural architecture, 1 for the deterministic reference),
ordered by the error metric the decision layer consumes. It states
plainly that every neural architecture loses to persistence, that the
tree is the only trained forecaster that clears it (+1.45 s MAE,
p_holm = 7e-4), and that the tree is therefore the framework's forecasting
component. It then explains *why* the reference is hard to beat from two
independent directions — the horizon is 300 s against a 92.5 s median
dwell, and 43.5% of the winning model's attribution sits on the last
observed value — so the small margin is accounted for rather than
excused.

**`significance.tex`** — the four design decisions first (the resampling
unit is never a row; a day drawn twice is re-keyed; every policy is
scored on the same resample; statistical and economic significance are
separate columns), then the results. Two things it is careful about:

- The decision table is presented as needing **two readings**. Row by
  row, no policy but the oracle demonstrates a saving; as paired
  contrasts on the same resample, proposed − static survives at
  p = 0.001. Both readings are printed rather than whichever is more
  favourable.
- The n = 4 motor-machine comparison is reported as **untestable by
  construction** — the signed-rank floor at n = 4 is 2/2⁴ = 0.125, above
  α — and in exactly those words, not as "no significant difference".

**`multimachine.tex`** — the eight-machine table carries the weekly-block
interval and the block count per row, so the milling machines' 5.5 h/day
cannot be read as the plant's largest opportunity. It states the two
mechanisms behind the auxiliary-machine failures (state names assigned by
mean active power can misname a component carrying standing reactive
current; the independent Otsu classifier drops to 55% exactly where the
model has nothing to lock onto), and reports the milling machines' −1,041
and −1,256 USD/yr as a deployment criterion rather than an anomaly.

**`crossmachine.tex`** — Parts A and B side by side, because the contrast
*is* the result: the duration forecaster transfers at a 4.8% cost in MAE
and the sequence model loses 86% of its STANDBY F1 across machine types,
and the attributions say why. Ends with the cross-dataset section,
including the negative control the battery nearly fails to reject.

### A discrepancy in the Phase IV notes, resolved

`PHASE4_NOTES.md`'s per-source currency table for pelletizer-I is the
**scaled** variant, while the sentence beside it about dpc-I's 8,695 s is
the **raw** variant. Both numbers are real; the table mixes them.
`crossmachine.tex` reports the **raw** variant throughout — no target
information used at all, which is the honest zero-shot condition — and
labels it as such. The narrative claims hold in raw: five of six
cross-type sources lose money on this target, and 5.6% of cross-type
pairs beat ski-rental against 33% of within-machine pairs. Under the
scaled variant exhaust-fan-I turns from +6.74 to −1.38 USD/yr, which is
the one cell that moves the "five of six" count, and it moves it in the
same direction.

---

## Task 14C — assembly, and what the checker is for

`generate_paper_draft.py` writes `paper/main.tex` from an ordered section
list and refuses to write it if any of six checks fail:

1. every `\ref`/`\eqref` resolves, and no label is defined twice
   (LaTeX silently keeps the last);
2. every `\cite` key resolves against `references.bib`;
3. every `\includegraphics` resolves on a graphics path;
4. no `XX`, `TODO` or `FIXME` survives in body text;
5. every table and figure is referred to by the text;
6. environments balance and every `tabular` row has as many fields as its
   column spec declares.

**Check 6 exists because there is no TeX distribution on this machine.**
`pdflatex`, `xelatex`, `latexmk` and `tectonic` are all absent, so the
draft cannot be compiled here and this is the honest limit of what has
been verified: the assembled document passes every check that can be run
on the source, and *that is not the same as building*. An unbalanced
environment or a wrong column count is the most likely way it fails on a
machine that does have LaTeX, so both are checked statically. The column
counter and the field splitter were unit-tested against hand-worked
specs (`*{4}{c}l` → 5, `\multicolumn{3}{c}{x & y} & b` → 4) and against
deliberate failures, then run on the real files.

The first run found **eight orphaned floats** — tables and figures no
sentence pointed at, which in a two-column layout drift somewhere
unhelpful and which a reader has no reason to look at. All eight now have
a reference.

### Assembly order, and why it differs from the phase order

```
Introduction → Related work → Method → State layer → Forecasting
  → Decision → Significance → Multi-machine → Cross-machine
  → Explainability → Ablation → Limitations → Conclusion
```

Three deliberate departures from the order in the plan. **Significance
follows both result sections it tests**, so a reader has seen the
forecasting table and the policy table before being shown what survives
resampling; putting it earlier would report intervals on policies not yet
defined. **Multi-machine follows the decision layer**, because its
headline result is that no other machine in the plant changes the
decision layer's conclusion. **Limitations precedes the conclusion**, so
the conclusion does not have to hedge — the boundaries are already
stated and it can be direct.

### Dangling references that were retargeted rather than left

- `sec:sustainability` (referenced from `decision_layer.tex`) →
  `sec:multimachine`. The sentence asked whether a machine with a larger
  energised-idle fraction changes the conclusion; Task 8 answers it, and
  the answer is no. The improvement plan's Phase VII sustainability
  analysis is **not in this paper**.
- `sec:method:hmm` and `sec:discussion` were referenced only from the
  superseded `novelty_and_contributions.tex`; the live text uses
  `sec:method:ghmm` and `sec:limitations`.
- `related_work.tex`'s "approximately 111 days" for IMDELD was softened
  to "a five-month window" with a pointer to the measured per-machine
  coverage, because the method section reports 63.4 covered days of a
  155-day span for pelletizer-I and the two figures read as a
  contradiction.

---

## Task 14D — limitations

Eight subsections, every one traceable to a measured result rather than a
generic caveat: the 18-day decision sample; the milling machines'
non-annualisability; the window forecaster's 20% voltage dependence;
three separate qualifications on the attributions (one machine, a
surrogate rather than the model, unmodelled feature dependence); what the
ablation does not cover; four hyperparameters stated rather than learned,
including the restart-cap anomaly where a *tighter* cap doubles the
saving; the scope of the evidence; and a closing subsection stating four
conditions under which the framework **should not be deployed**, two of
which it detects itself and two of which it does not.

The one Phase VI item the plan listed that is now stated as a limitation
rather than a fix: **the physics battery is not a row of the ablation
table**, because it is measured over labelling seeds while every other
row is measured over held-out blocks, windows or days. That is the
strongest accept/reject result in the project and its absence is
acknowledged in the section that prices components and again in the
limitations.

---

## Task 14E — abstract and introduction

Four things changed, each because a measurement forced it.

**The forecasting component is a boosted tree.** Every description of the
Seq2Seq LSTM as the proposed forecaster is gone; the sequence models are
described as compared baselines and the negative result is stated in the
abstract, not buried.

**The decision-layer figure is +2.9 ± 0.6 USD/yr and is not claimed as a
saving.** The abstract gives the point estimate with its interval
`[-56, +46]` and immediately gives the claim that does survive: +68.9
USD/yr over the fixed-threshold rule at p = 0.001. An abstract that
quoted +2.9 without the interval would be quoting noise to one decimal
place.

**The calendar share is 82%.**

**The accept/reject loop is contribution 1**, ahead of the temporal
model, and the abstract carries its evidence: 3 seeds in 10 place the
standby cluster on a physically different state, and conditioning on a
passing score collapses the spread from ±33.7% to ±3.0%.

**Every `XX` placeholder is gone**, and the checker enforces that it
stays gone. The Phase I abstract had five; the Phase III note warning
that the payback placeholder could not be filled with a favourable figure
is resolved by not making payback a headline at all — the abstract's
closing claim is that the framework's demonstrated value on this facility
is decision quality and validated state inference, not recovered energy.

The forward references to a sustainability section and a deployment
section were **removed rather than left dangling**. Promising analyses in
a contribution list and not delivering them is worse than not promising
them.

---

---

## Task 14F — closing the four gaps left open above

The four items this note originally listed as outstanding are now
three-and-a-bit closed. What follows replaces the "what is not done"
list as it stood; the residue is at the bottom and is real.

### The five table-only results now have figures

`experiments/make_figures_phase7.py` renders them from the same stored
outputs the tables are typeset from — it fits nothing and refits
nothing — at the Phase VI conventions: IEEE column widths placed at
100%, no suptitle, PDF **and** PNG, `pdf.fonttype = 42`, and nothing
encoded by hue alone.

| Figure | Carries | Section |
|---|---|---|
| `fig_p7_state_detection` | flicker vs dwell, decoding path as a trajectory; STANDBY hours exposing spectral | `state_layer.tex` §4.2 |
| `fig_p7_min_dwell` | the sweep: flicker collapses, nothing else moves | `state_layer.tex` §4.3 |
| `fig_p7_economic_plane` | break-even, saving, energy, with the zero contour | `decision_layer.tex` §6.6 |
| `fig_p7_transfer` | both 8×8 matrices on one axis convention | `crossmachine.tex` §9.1–9.2 |
| `fig_p7_cross_dataset` | the physics battery check-by-check, and what it fell back on | `crossmachine.tex` §9.3 |

Each is drawn to carry the claim the surrounding prose makes rather
than to reproduce the table. Three are worth naming:

- **State detection is a trajectory, not a bar chart.** The three
  decoding configurations share one set of GMM emissions and differ
  only in how they are decoded, so drawing them as connected points
  over flicker × dwell makes "the emission model barely matters, the
  decoding rule matters entirely" a shape rather than a sentence.
- **The two transfer matrices are one figure.** §9.1 and §9.2 argue a
  *contrast*, and the contrast only lands side by side: panel A varies
  down its columns and barely across its rows, panel B is a bright
  diagonal on a dark field.
- **The cross-dataset figure is a status grid, not scores.** The
  finding is *which checks were unavailable*, and a bar chart of the
  surviving scores hides precisely that. C3 and C4 are boxed because
  they are the only two independent of power magnitude.

Two rendering bugs were found and fixed while checking the output, both
of which would have shipped silently: `subplots_adjust` called after
`fig.colorbar(..., ax=ax)` moves the panels out from under their own
colorbars (now `constrained_layout`), and a short vertical colorbar
label sits close enough to the next panel to read as *that* panel's
y-axis label (units moved into the titles).

### The bibliography is verified

`experiments/verify_bibliography.py` resolves every DOI against
Crossref — with a DataCite fallback, because the IEEE DataPort dataset
deposit is not in Crossref and 404s there — and compares title, year,
volume, issue and pages field by field. **47 of 47 now match; it exits
non-zero on any mismatch or unresolvable DOI.**

Five entries were wrong:

| Entry | What was wrong |
|---|---|
| `holmegaard2016industrial` | **the DOI resolved to a different paper** (…7501697 is Guo & Singh on IoT service levels; the right one is …7501709) |
| `imdeld2018dataset` | **wrong author list** — named J. G. R. C. Gomes, who authored the *paper* but did not deposit the *dataset*, and omitted three depositors |
| `campello2015hdbscan` | page range (`5:1--5:51` → `1--51` + `articleno`) |
| `karlin1988competitive` | issue (`1` → `1--4`) |
| `albers2003online` | issue (`1--2` → `1`) |

and three gained a DOI they should always have carried
(`dempster1977em`, `zhang2018seq2point`, `efron1993bootstrap`).

The first two are the ones that mattered: a DOI pointing at someone
else's paper and a misattributed dataset are both errors a referee
finds in one click, and neither was among the fields the `% VERIFY`
markers flagged as suspect. **The markers were pointing at the wrong
things** — they warned about volume and page numbers, which were almost
all correct, and said nothing about the two entries that were actually
broken. That is an argument for checking mechanically rather than for
checking what you remember being unsure about.

Four apparent mismatches were the checker's fault, not the bib's, and
each needed a rule: Crossref titles carry JATS markup (`<i>EM</i>`
normalises to "i em i" unless tags are stripped first); ACM deposits
the subtitle separately, so "XGBoost: A Scalable Tree Boosting System"
is deposited as the bare word "XGBoost"; issue *ranges* are `18--19` in
BibTeX and `18-19` at Crossref; and a print/online year split of one
year is routine. Authors are compared on surname sets and reported as
a note, never a mismatch, because accent deposition is too inconsistent
to fail on.

`dempster1977em` is worth one line: the range everyone cites is
`1--38`, which includes the printed discussion. The publisher record
for the paper itself is `1--22`, and that is what the DOI shows.

### Front and back matter, and the venue switch

Both went into `generate_paper_draft.py`, not into `main.tex` — the
file says "GENERATED … do not edit by hand" at the top and the next run
would have discarded them. The preamble now carries an author block, and
the postamble an acknowledgements, data-availability and
competing-interest section, all before the bibliography where every
target venue puts them. The preamble also documents the exact three-line
change for IEEEtran, elsarticle and Springer, including the two that
need the author block restructured rather than merely filled.

The generated front and back matter is now **checked like a section
file** — it carries three `\ref`s and a `\cite`, and a dangling one
there breaks the build exactly as it would in the body.

Nine `FILL_IN` markers remain, and they are **counted and listed on
every run but do not fail the check**. An unfilled author list is an
incomplete submission, not an inconsistent document, and refusing to
assemble over it would stop the draft being read. They are: author name,
affiliation and email; funding; the competing-interest declaration; and
the licence and provenance of the two second-site records. None can be
filled from the repository.

## The draft as it stands

23,805 words of body text, 112 labels, 174 cross-references, 99
citations over 63 bibliography entries, **15 figures**, 18 table floats
(plus 4 small inline tabulars). Thirteen sections and 53 subsections,
plus abstract, acknowledgements, data availability and competing
interests.

## What is still not done

- **It has not been compiled.** No TeX on this machine. Six static
  checks pass; that is a weaker statement and is made as one. First
  build elsewhere should be expected to surface float placement and
  column-width issues, particularly the four `figure*` environments
  added in this phase, the two `table*` environments, and the
  eight-column related-work table.
- **Nine `FILL_IN` markers**, listed above. These are the only thing
  between the current draft and a submittable one.
- **One bibliography entry cannot be machine-checked.**
  `gutowski2006electrical` — the 13th CIRP LCE proceedings were never
  assigned DOIs. Author, title, venue and year are certain; only the
  page range is from secondary sources. It carries the sole remaining
  `% VERIFY`. Fifteen further entries have no DOI because none was ever
  registered (ISO/IEC standards, JMLR, and the NeurIPS/AAAI/KDD/ICLR
  proceedings of their era); those fields are not in doubt.
- **No sustainability or deployment analysis.** Phases VII and VIII of
  the improvement plan (kWh/CO₂/ROI quantification; edge-deployment
  latency and architecture) are not in this draft. The economic material
  that does exist is in the decision layer and the multi-machine section,
  and the facility total — under 90 USD/yr — is what a sustainability
  section would have to start from.
