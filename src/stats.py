"""
============================================================
STATISTICAL TOOLKIT  --  src/stats.py
============================================================

PURPOSE
-------
Phase V.  Every number in Phases I-IV is a point estimate from a
single run.  This module supplies the machinery to say which of the
differences between those numbers are real and which are noise, and
it is deliberately opinionated about four things that are easy to
get wrong on this data.

1.  THE RESAMPLING UNIT IS NOT A ROW
    A 1 Hz electrical record is massively autocorrelated: consecutive
    seconds are the same machine in the same state.  An i.i.d.
    bootstrap over rows treats 5.5 million readings as 5.5 million
    independent observations and returns confidence intervals that
    are too narrow by orders of magnitude.  `moving_block_bootstrap`
    resamples contiguous blocks instead, and `group_bootstrap`
    resamples whole natural units -- idle episodes, or factory days.
    Block length is a stated choice with a stated cost: too short
    destroys the dependence the interval is supposed to respect, too
    long leaves too few blocks to resample.

2.  PAIRED TESTS, NOT UNPAIRED ONES
    Two forecasters scored on the SAME test windows produce paired
    errors, and the pairing removes the window-to-window variance
    that dominates the unpaired comparison.  `wilcoxon_paired` is
    therefore the default, and it returns an effect size and a
    Hodges-Lehmann location shift alongside the p-value, because a
    p-value on 5,449 paired windows will be small for differences far
    too tiny to act on.

3.  FAMILY-WISE ERROR CONTROL WITHOUT BONFERRONI
    With 7 models the pairwise family is 21 tests; plain Bonferroni
    at alpha/21 is needlessly conservative.  `holm` controls the same
    family-wise error rate uniformly more powerfully, and
    `benjamini_hochberg` controls the false discovery rate for the
    exploratory comparisons.  Both are reported next to the raw
    p-values so a reader can see what the correction cost.

4.  A NON-SIGNIFICANT RESULT IS NOT AN EQUIVALENCE RESULT
    With four viable machines the Friedman test has very little
    power, and failing to reject tells you almost nothing.
    `friedman_power` measures how little, by simulation, so the
    paper can state the detectable effect size instead of implying
    that models are equivalent.

CITATIONS
---------
Wilcoxon (1945); Friedman (1937); Demsar (2006) JMLR 7:1-30 for the
Friedman/Nemenyi protocol and the critical-difference diagram;
Kunsch (1989) for the moving block bootstrap; Holm (1979);
Benjamini & Hochberg (1995); Hodges & Lehmann (1963).
============================================================
"""

from __future__ import annotations

import numpy as np

# Studentised range statistic q_alpha / sqrt(2) for the Nemenyi test, indexed
# by the number of treatments k (Demsar 2006, Table 5).  Tabulated rather than
# computed because the exact distribution requires numerical integration and
# the table is the standard reference every comparable paper cites.
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
           8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030},
}


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def moving_block_bootstrap(
    x: np.ndarray,
    statistic,
    block_len: int,
    n_boot: int = 2000,
    random_state: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Percentile bootstrap for a statistic of an autocorrelated series.

    Contiguous blocks of `block_len` are drawn with replacement and
    concatenated to a series of the original length (Kunsch 1989), so the
    within-block dependence survives resampling and only the between-block
    dependence is destroyed.  For a 1 Hz industrial record the dependence that
    matters is the duty cycle, which is why the block length used in Phase V is
    one week rather than one minute.

    Returns the point estimate, the percentile interval, and the number of
    effective blocks -- the last so that a degenerate resample (a handful of
    blocks) is visible rather than hidden inside a suspiciously tight interval.
    """
    x = np.asarray(x)
    n = len(x)
    block_len = int(max(1, min(block_len, n)))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(random_state)

    point = float(statistic(x))
    starts_max = max(1, n - block_len + 1)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        draws[b] = statistic(x[np.minimum(idx, n - 1)])

    lo, hi = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(np.std(draws, ddof=1)),
        "n_boot": int(n_boot),
        "block_len": block_len,
        "n_blocks": n_blocks,
        "alpha": alpha,
    }


def block_mean_bootstrap(
    x: np.ndarray,
    block_len: int,
    n_boot: int = 2000,
    random_state: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Moving-block bootstrap specialised to the MEAN, computed in closed form.

    For the mean, a resample's value is the average of its blocks' means, so
    the block means can be read off a cumulative sum instead of materialising
    the resampled series.  On a 5.5-million-row record with one-week blocks
    that is the difference between a 44 MB index array per draw and three
    array lookups, and the two are numerically identical up to the truncation
    of the final partial block.

    Used for state-time fractions (the STANDBY share of readings), where the
    statistic is a mean by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    block_len = int(max(1, min(block_len, n)))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(random_state)

    cs = np.concatenate([[0.0], np.cumsum(x)])
    starts_max = max(1, n - block_len + 1)
    starts = rng.integers(0, starts_max, size=(n_boot, n_blocks))
    block_means = (cs[starts + block_len] - cs[starts]) / block_len
    draws = block_means.mean(axis=1)

    lo, hi = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "point": float(x.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(np.std(draws, ddof=1)),
        "n_boot": int(n_boot),
        "block_len": block_len,
        "n_blocks": n_blocks,
        "alpha": alpha,
    }


def group_bootstrap(
    groups: np.ndarray,
    statistic,
    n_boot: int = 2000,
    random_state: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap that resamples whole GROUPS with replacement.

    `groups` is an array of group labels, one per observation; `statistic`
    receives the array of positional indices selected on each draw and returns
    a scalar.  This is the right resampling unit whenever the observations
    cluster: idle episodes for the decision layer (the episode is what the
    policy acts on), or factory days when the objective carries a per-day
    constraint -- resampling episodes independently would break the daily
    restart cap that the evaluation enforces.

    The group count, not the observation count, is what determines the width of
    the interval, and it is returned so that the reader can see it.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(random_state)

    point = float(statistic(np.arange(len(groups))))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in picked])
        draws[b] = statistic(idx)

    lo, hi = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(np.std(draws, ddof=1)),
        "n_groups": int(len(uniq)),
        "n_boot": int(n_boot),
        "alpha": alpha,
        "p_two_sided_vs_zero": float(
            2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def bootstrap_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(draws), [alpha / 2 * 100,
                                               (1 - alpha / 2) * 100])
    return float(lo), float(hi)


# ── Paired tests and effect sizes ─────────────────────────────────────────────

def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided",
                    hl_max_pairs: int = 4_000_000, random_state: int = 0) -> dict:
    """
    Wilcoxon signed-rank test on paired observations, with effect size.

    Returns, besides the p-value:

    `rank_biserial`   (W+ - W-) / (W+ + W-), the matched-pairs effect size:
                      +1 means every non-tied pair favours `a`, 0 means the
                      signed ranks cancel.  Reported because a p-value computed
                      over thousands of paired windows says only that a
                      difference exists, not that it is large enough to matter.
    `hl_shift`        the Hodges-Lehmann estimator -- the median of the Walsh
                      averages (d_i + d_j)/2 -- which is the location shift the
                      signed-rank test is actually testing, in the units of the
                      data.  Subsampled when the pair count would exceed
                      `hl_max_pairs`.
    `median_diff`     the plain median of the differences, for readers who want
                      the simpler summary.
    `n_nonzero`       pairs that contribute; all-tied pairs carry no
                      information and their exclusion is reported.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length inputs")
    d = a - b
    nz = d[d != 0]
    out = {
        "n": int(len(d)),
        "n_nonzero": int(len(nz)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
    }
    if len(nz) == 0:
        out.update({"statistic": float("nan"), "p_value": 1.0,
                    "rank_biserial": 0.0, "hl_shift": 0.0,
                    "note": "all pairs tied"})
        return out

    stat, p = _wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    from scipy.stats import rankdata
    r = rankdata(np.abs(nz))
    w_plus = float(r[nz > 0].sum())
    w_minus = float(r[nz < 0].sum())

    rng = np.random.default_rng(random_state)
    if len(nz) ** 2 > hl_max_pairs:
        m = int(np.sqrt(hl_max_pairs))
        s = rng.choice(nz, size=min(m, len(nz)), replace=False)
    else:
        s = nz
    walsh = (s[:, None] + s[None, :]) / 2.0
    iu = np.triu_indices(len(s))
    out.update({
        "statistic": float(stat),
        "p_value": float(p),
        "rank_biserial": float((w_plus - w_minus) / (w_plus + w_minus)),
        "hl_shift": float(np.median(walsh[iu])),
        "alternative": alternative,
    })
    return out


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference over its own SD."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples, pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


# ── Multi-model comparison ────────────────────────────────────────────────────

def friedman_test(scores: np.ndarray, lower_is_better: bool = True) -> dict:
    """
    Friedman test over a (blocks x treatments) score matrix, with Kendall's W.

    Blocks are whatever the comparison is repeated over -- seeds, machines,
    folds -- and every treatment must be scored on every block.  Kendall's W
    (= chi2_F / (N(k-1))) is returned as the effect size: it is the fraction of
    the maximum possible rank concordance, so W near 0 means the models are
    ordered differently on every block and W near 1 means the ordering is
    identical everywhere.  A significant Friedman with tiny W is a real but
    inconsistent difference, which is a different claim from a large one.
    """
    from scipy.stats import friedmanchisquare, rankdata

    S = np.asarray(scores, float)
    n, k = S.shape
    if k < 3:
        raise ValueError(
            "the Friedman test is undefined for fewer than three treatments; "
            "for a two-treatment comparison use wilcoxon_paired, which is the "
            "test the Friedman procedure reduces to")
    if n < 2:
        raise ValueError("Friedman needs at least 2 blocks")
    signed = S if lower_is_better else -S
    ranks = np.vstack([rankdata(row) for row in signed])
    mean_ranks = ranks.mean(axis=0)

    stat, p = friedmanchisquare(*[S[:, j] for j in range(k)])
    return {
        "n_blocks": int(n),
        "k_treatments": int(k),
        "statistic": float(stat),
        "p_value": float(p),
        "mean_ranks": mean_ranks.tolist(),
        "kendalls_w": float(stat / (n * (k - 1))),
        "lower_is_better": lower_is_better,
    }


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """
    Critical difference for the Nemenyi post-hoc test (Demsar 2006, eq. 6):

        CD = q_alpha * sqrt( k(k+1) / (6N) )

    Two treatments differ significantly if their mean ranks differ by more than
    CD.  Note what the formula says about power: CD shrinks only as sqrt(N), so
    halving the detectable rank gap costs four times the blocks.
    """
    q = NEMENYI_Q.get(alpha, NEMENYI_Q[0.05])
    if k not in q:
        raise ValueError(f"no tabulated q for k={k} at alpha={alpha}")
    return float(q[k] * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def nemenyi_matrix(mean_ranks, k: int, n_blocks: int, alpha: float = 0.05):
    """Pairwise |rank difference| and whether it exceeds the critical difference."""
    mr = np.asarray(mean_ranks, float)
    cd = nemenyi_cd(k, n_blocks, alpha)
    diff = np.abs(mr[:, None] - mr[None, :])
    return diff, diff > cd, cd


# ── Multiple-comparison corrections ───────────────────────────────────────────

def holm(pvals) -> np.ndarray:
    """
    Holm (1979) step-down adjustment: controls the family-wise error rate and
    is uniformly at least as powerful as Bonferroni, which is why Phase V uses
    it for the confirmatory family.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) step-up adjustment: controls the false discovery
    rate rather than the family-wise error rate, which is the appropriate
    target for the exploratory families here (every model pair on every
    machine) where a few false positives are tolerable and missing real effects
    is not.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        running = min(running, p[idx] * m / (i + 1))
        adj[idx] = min(1.0, running)
    return adj


# ── Power ─────────────────────────────────────────────────────────────────────

def friedman_power(
    n_blocks: int,
    k: int,
    effect: float,
    n_sim: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> float:
    """
    Simulated power of the Friedman test to detect a shift of `effect`
    standard deviations in one treatment out of `k`, with `n_blocks` blocks.

    Phase V uses this to put a number on a limitation rather than merely
    conceding it: with four motor machines, a non-significant Friedman is
    almost uninformative, and the honest statement is "we can detect an effect
    of size X with power Y", not "the models are equivalent".
    """
    if k < 3:
        raise ValueError("use wilcoxon_power for a two-treatment comparison")
    rng = np.random.default_rng(random_state)
    from scipy.stats import friedmanchisquare

    hits = 0
    for _ in range(n_sim):
        block = rng.normal(0, 1, size=(n_blocks, 1))       # block effect
        noise = rng.normal(0, 1, size=(n_blocks, k))
        shift = np.zeros(k)
        shift[0] = effect
        S = block + noise + shift
        _, p = friedmanchisquare(*[S[:, j] for j in range(k)])
        hits += int(p < alpha)
    return hits / n_sim


def wilcoxon_power(
    n_pairs: int,
    effect: float,
    n_sim: int = 4000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> float:
    """
    Simulated power of the paired Wilcoxon signed-rank test to detect a mean
    paired difference of `effect` standard deviations, with `n_pairs` pairs.

    This is the companion to `friedman_power` for the two-treatment case that
    Phase V actually faces most often: one proposed model against one baseline,
    paired over a handful of machines.  At n = 8 the test cannot return a
    p-value below 0.0078 no matter how large the effect, because that is the
    probability of the most extreme rank configuration -- a hard floor worth
    stating before any conclusion is drawn from it.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    rng = np.random.default_rng(random_state)
    hits = 0
    for _ in range(n_sim):
        d = rng.normal(effect, 1.0, size=n_pairs)
        if np.all(d == 0):
            continue
        try:
            _, p = _wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            continue
        hits += int(p < alpha)
    return hits / n_sim


def wilcoxon_p_floor(n_pairs: int) -> float:
    """
    Smallest two-sided p-value the signed-rank test can return with `n_pairs`
    pairs: 2 / 2^n.  Reported alongside every small-n test so a
    "non-significant" verdict cannot be mistaken for evidence of equivalence.
    """
    return float(min(1.0, 2.0 / (2 ** n_pairs)))


def min_detectable_effect(
    n_blocks: int, k: int, target_power: float = 0.80,
    grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0), **kw
) -> dict:
    """Smallest simulated effect on the grid reaching `target_power`."""
    curve = {float(e): friedman_power(n_blocks, k, e, **kw) for e in grid}
    reached = [e for e, p in curve.items() if p >= target_power]
    return {"power_curve": curve,
            "min_detectable_effect": (min(reached) if reached else None),
            "target_power": target_power,
            "n_blocks": n_blocks, "k": k}


# ── Reporting helpers ─────────────────────────────────────────────────────────

def stars(p: float) -> str:
    """Conventional significance marker; never used without an effect size."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def describe_runs(values, alpha: float = 0.05) -> dict:
    """mean +/- sd over repeated runs, with the min and max actually observed."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    out = {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
    }
    if len(v) > 1:
        from scipy.stats import t as _t
        se = out["std"] / np.sqrt(len(v))
        h = se * _t.ppf(1 - alpha / 2, len(v) - 1)
        out["ci_lo"], out["ci_hi"] = out["mean"] - h, out["mean"] + h
    return out
