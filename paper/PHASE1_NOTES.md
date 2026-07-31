# Phase I — Scientific Foundation: completion notes

Status: **complete**. Covers Task 1 (restructure literature review) and
Task 2 (redefine novelty). No experiments or code changes were required by
this phase.

## Files produced

| File | Contents | Goes into |
|---|---|---|
| `references.bib` | 51 BibTeX entries, grouped by theme A–E | bibliography |
| `related_work.tex` | Section 2: four-theme review, comparison table, gap statement | Section 2 |
| `novelty_and_contributions.tex` | reframed abstract, positioning statement, contribution list, organisation paragraph | Abstract + end of Section 1 |
| `PHASE1_NOTES.md` | this file | not published |

Nothing that already existed was modified. The `.docx` reports in `outputs/`
are untouched; `related_work.tex` supersedes the two-paragraph, zero-citation
"Related Work & Background" section of `outputs/Research_Report.docx`.

## Task 1 — literature review restructure

Organised into the four themes the plan specified, with the intended
argumentative shape: each theme ends with a paragraph naming what that
community *cannot* do, so the gap statement in §2.5 is a summary rather than
an assertion.

- **§2.1 Machine State Identification** — supervised vs unsupervised;
  k-means / GMM / DBSCAN / HDBSCAN / spectral; the i.i.d. limitation and why
  HMM resolves it; sticky prior; Otsu for threshold-free cut-offs.
  Opens with ISO 14955-1 as the *definitional* anchor for "standby", which is
  the citation the plan asked to be used throughout.
- **§2.2 Non-Intrusive Load Monitoring** — Hart → probabilistic → deep;
  benchmarks and evaluation protocols; then the two decisive points:
  the field is residential (with Holmegaard & Kjærgaard as the evidence that
  residential algorithms degrade industrially), and its objective is
  *attribution*, not forecasting or actuation. IMDELD is introduced here,
  including the fact that its authors supply no per-timestamp annotation —
  which is what forces the label-free design and the physics validation.
- **§2.3 Industrial Energy Optimisation** — Gutowski / Dahmus / Kara & Li /
  Devoldere / Duflou on idle energy dominance; ISO 50001 and ISO 14955-1 on
  the management side. Closes on the methodological limitation: laboratory
  measurement of single processes yields design guidance, not online
  inference for a running facility.
- **§2.4 Predictive Shutdown and Decision Support** — Mouzon's break-even
  trade-off as the canonical formulation, then Chen / Sun & Li / Li & Sun /
  Frigerio & Matta / He. Closes on the universal assumption that the idle
  duration is *given*.

**Comparison table** (`\label{tab:related}`): 17 works × 5 capability
criteria (labels required, temporal model, duration forecast, economic
decision, real industrial data) + 2 descriptive columns, grouped by theme,
with "This work" as the final row. Two footnotes qualify the entries that
would otherwise overstate: Hart is unsupervised in clustering but needs a
curated signature library; Frigerio & Matta model idle duration
stochastically but do not forecast it from measurements.

Set as a `table*` (full page width). Preamble needs `booktabs`,
`threeparttable`, `multirow`, `wasysym`. Column count was verified
programmatically (8 columns, all 21 rows consistent) and every `\cite` key
was checked to resolve against `references.bib` — no undefined references.

**Gap statement** (§2.5): boxed, one paragraph, phrased as the *non-existence
of a connection* rather than the absence of a method — "the former produces
labels no controller consumes, the latter consumes durations no inference
procedure supplies." Followed by a paragraph explaining why the three
components cannot simply be concatenated (no ground truth to validate
against; flicker makes duration forecasting ill-posed; a point forecast
carries uncertainty a fixed threshold ignores). That paragraph is what
justifies the framework as a methodology contribution rather than an
integration exercise.

## Task 2 — novelty reframing

The plan's replacement text is used essentially verbatim as the three
numbered components, with three additions:

1. **A reframed abstract**, because the old framing ("we apply GMM and
   LSTM") also lives in the abstract, and that is what a reviewer reads
   first. Numeric placeholders are marked `XX` — fill from the final run.
2. **An explicit scope-of-claim paragraph.** We do *not* claim novelty in
   the Gaussian HMM, the Seq2Seq LSTM, or the break-even criterion. Stating
   this before a reviewer does converts the paper's weakest attack surface
   into evidence of self-awareness; without it, "these are all off-the-shelf
   components" is the obvious desk-reject line.
3. **A contribution list where each item names the experiment that
   substantiates it**, with forward `\ref`s. Contributions a reviewer cannot
   trace to a table are read as claims, not contributions.

## One correction to the plan's premises

The plan's gap table lists *"HMM smoothing — planned but not completed"*.
This is out of date. `validate_gmm.py:440` `fit_gaussian_hmm()` fits a full
`hmmlearn.GaussianHMM` by Baum–Welch with a sticky self-transition prior
(Fox et al. 2011) and decodes with Viterbi; per the docstring at
`validate_gmm.py:443–451` it **replaced** the earlier bolted-on
`viterbi_smooth()` + dwell-time grid search. The legacy independent smoother
survives only at `src/data_analysis.py:117`.

This matters for Phase I specifically: it is why the "This work" row of
Table 1 can honestly claim `\CIRCLE` under *temporal model*, and why the
framework is described as *jointly* learning emissions and transitions
rather than clustering-then-smoothing. That joint framing is a stronger
claim and is the correct description of the code.

Phase IX's ablation should therefore be **GMM (i.i.d.) vs GMM-HMM (joint)**,
not "GMM vs GMM + smoothing" — the ablation baseline needs `fit_gmm()` at
`validate_gmm.py:401`, which is still present for exactly this purpose.

## Citation verification — required before submission

Entries were written from domain knowledge. Titles, authors, venues and
years are reliable; **volume/issue/page/DOI fields reconstructed from memory
can drift**, and a wrong page range is a referee-visible error. Every entry
needing confirmation carries a `% VERIFY` comment in `references.bib`.

Verify these against Crossref/DOI resolution:

| Key | Field(s) to confirm |
|---|---|
| `campello2015hdbscan` | volume, article number |
| `bonfigli2018denoising` | volume, pages |
| `holmegaard2016industrial` | DOI, pages |
| `martins2018imdeld` | full author list, DOI |
| `schirmer2023nilmreview` | volume, pages, DOI |
| `gutowski2006electrical` | pages |
| `dahmus2004environmental` | pages, DOI |
| `devoldere2007improvement` | it is a book chapter — check series/publisher |
| `mouzon2008framework` | volume, pages |
| `chen2013schedule` | DOI |
| `sun2013opportunity` | DOI |
| `frigerio2015energy` | DOI |
| `he2015energy` | volume, pages |
| `li2016energy` | volume, pages, DOI — note the key says 2016 but the year field is 2013; reconcile |

Two further checks:

- **`mouzon2007operational`** — the plan cited Mouzon et al. (2007) under the
  title *"Contribution of individual machine idle energy consumption towards
  total manufacturing energy consumption."* The bib uses *"Operational
  methods for minimization of energy consumption of manufacturing
  equipment"* (IJPR 45(18–19)), which is the paper that actually establishes
  the break-even trade-off cited in §2.4. If the other title is a distinct
  paper you intend to cite, add it separately.
- **`ester1996dbscan`** — the plan lists the venue as AAAI. It is KDD-96;
  AAAI Press is the publisher. The bib records it correctly.

`grep -n "VERIFY" paper/references.bib` lists all flagged entries.

## Carry-forward to later phases

- `references.bib` already contains the method citations for Phases II
  (TCN, PatchTST, GRU, XGBoost), V (Wilcoxon, Friedman, Demšar), VI (SHAP)
  and VIII (IEC 61131-3), so those phases add prose, not bibliography.
- `novelty_and_contributions.tex` forward-references
  `\label{tab:ablation}`, `\ref{sec:method:hmm}`, `\ref{sec:validation}`,
  `\ref{sec:forecasting}`, `\ref{sec:decision}`, `\ref{sec:results}`,
  `\ref{sec:sustainability}`, `\ref{sec:discussion}`,
  `\ref{sec:conclusion}`. These will produce `??` until the corresponding
  phases define them. That is expected and is the intended checklist.
- Abstract `XX` placeholders: flicker rate before/after (Phase IX),
  standby-duration MAE (Phase II), kWh/year and payback (Phase VII).
