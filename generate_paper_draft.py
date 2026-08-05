"""
============================================================
PAPER DRAFT ASSEMBLY  --  generate_paper_draft.py
============================================================

PURPOSE
-------
Assemble the section files under `paper/` into a single compilable
LaTeX document (`paper/main.tex`), and -- more usefully -- refuse to do
so silently when the draft is internally inconsistent.

The sections were written one phase at a time, each forward-referencing
labels that later phases would define.  That is a deliberate working
convention, but it means the only way to know whether the draft is
whole is to check it mechanically.  This script therefore runs four
checks and reports every failure before writing anything:

  1. CROSS-REFERENCES   every \\ref / \\eqref resolves to a \\label that
                        some included file defines, and no label is
                        defined twice (LaTeX silently keeps the last).
  2. CITATIONS          every \\cite key resolves against references.bib.
  3. FIGURES            every \\includegraphics resolves to a file on one
                        of the graphics paths.
  4. PLACEHOLDERS       no `XX`, `TODO` or `FIXME` survives in body text.
  5. ORPHANED FLOATS    every table and figure is referred to by the text.
  6. STRUCTURE          environments balance, and every `tabular` row has
                        as many fields as the column spec declares.

Exit status is 0 only if all six pass.  `--force` writes the document
anyway, for inspecting an incomplete draft.

USAGE
-----
    python generate_paper_draft.py            # check, then write
    python generate_paper_draft.py --check    # check only
    python generate_paper_draft.py --force    # write despite failures

OUTPUT
------
    paper/main.tex

WHAT IS *NOT* DONE HERE
-----------------------
No LaTeX is compiled: there is no TeX distribution on this machine, and
a checker that depended on one would not run at all.  Checks 1-4 are the
subset of `latex`+`bibtex` diagnostics that can be performed on the
source, and they cover the failure modes a phased draft actually
produces (dangling \\ref, a citation added to prose but not to the bib,
a figure referenced by a name the run never wrote).  Check 6 exists
*because* nothing compiles here: an unbalanced environment or a tabular
row with the wrong number of fields is the most likely way this document
fails to build on a machine that has pdflatex, and it is cheap to detect
statically.  Passing all six is not the same as building; it is what can
be established without a compiler.
============================================================
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
BIB = PAPER / "references.bib"
OUT = PAPER / "main.tex"

# Generated tables live with the run that produced them, not in paper/.
ABLATION_TABLE = "../outputs/phase6/task13e/pelletizer-I/ablation_table"

# Searched, in order, for \includegraphics targets.  Mirrors the
# \graphicspath written into the preamble.
GRAPHICS_PATHS = [
    ROOT / "outputs" / "phase7" / "figures",
    ROOT / "outputs" / "phase6" / "figures",
    ROOT / "outputs" / "phase5" / "figures",
    ROOT / "outputs" / "phase4" / "figures",
    ROOT / "outputs" / "phase3" / "figures" / "pelletizer-I",
    ROOT / "outputs" / "phase2" / "figures" / "pelletizer-I",
]
GRAPHICS_EXTS = [".pdf", ".png", ".jpg", ".eps", ""]

# ---------------------------------------------------------------------
# The document.  Order is the argumentative order, which differs from
# the order the phases were run in:
#
#   * the statistical analysis follows BOTH result sections it tests,
#     so a reader has seen the forecasting table and the policy table
#     before being shown what survives resampling;
#   * the multi-machine study follows the decision layer, because its
#     headline result is that no other machine changes the decision
#     layer's conclusion;
#   * limitations precede the conclusion, so the conclusion does not
#     have to hedge.
# ---------------------------------------------------------------------
SECTIONS: list[tuple[str, str]] = [
    ("introduction",   "Introduction, positioning, contributions"),
    ("related_work",   "Four-theme review and the research gap"),
    ("method",         "Framework, data, state inference, the two forecasters"),
    ("state_layer",    "State-layer results, flicker, validation"),
    ("forecasting",    "Forecasting comparison against an untrained reference"),
    ("decision_layer", "Optimisation-based decision support"),
    ("significance",   "Significance testing and repeated runs"),
    ("multimachine",   "The eight machines"),
    ("crossmachine",   "Transfer between machines and between sites"),
    ("explainability", "Shapley attribution and transition structure"),
    ("ablation",       "Component ablation"),
    ("limitations",    "Limitations"),
    ("conclusion",     "Conclusion"),
]

# Files pulled in by another section rather than by main.tex directly.
NESTED = {"method_gmm_labeller": "method"}

# Retained for its design rationale; contains no body text.
NOT_INCLUDED = {"novelty_and_contributions"}

PREAMBLE = r"""% =====================================================================
%  main.tex  --  GENERATED by generate_paper_draft.py.  Do not edit by
%  hand: edit the section file named in the \input and re-run.
%
%  Target venue formatting is not fixed here.  The document class below
%  is a neutral placeholder that compiles; switching to IEEEtran or
%  elsarticle requires changing only this preamble, because no section
%  file uses a class-specific command except \CIRCLE/\LEFTcircle/\Circle
%  in the related-work table (wasysym) and table* in two places.
%
%  ------------------------------------------------------------------
%  SWITCHING VENUE CLASS.  Three lines change; no section file changes.
%
%    IEEE Transactions / IEEE Access
%      1. \documentclass[journal]{IEEEtran}
%      2. \bibliographystyle{IEEEtran}
%      3. delete \usepackage{lmodern} and the geometry line -- IEEEtran
%         sets its own margins and font, and geometry fights it.
%      Note: IEEEtran wants \IEEEauthorblockN/\IEEEauthorblockA inside
%      \author, so the author block below is rewritten, not just filled.
%
%    Elsevier (Applied Energy, Energy)
%      1. \documentclass[preprint,12pt]{elsarticle}
%      2. \bibliographystyle{elsarticle-num}
%      3. delete the geometry line; wrap the author block in
%         \begin{frontmatter}...\end{frontmatter} with \author and
%         \affiliation, and move \input{abstract} inside it.
%
%    Springer (SCIS, LNCS)
%      1. \documentclass{sn-jnl}  (or {llncs})
%      2. \bibliographystyle{sn-mathphys}
%      3. delete geometry and lmodern.
%
%  In every case the figures are already sized for a 3.5in single
%  column and a 7.16in double column, so nothing needs re-rendering.
%  ------------------------------------------------------------------
% =====================================================================
\documentclass[10pt,twocolumn]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{multirow}
\usepackage[nointegrals]{wasysym}   % \CIRCLE \LEFTcircle \Circle
\usepackage{graphicx}
\usepackage{amsthm}
\usepackage[hidelinks]{hyperref}

\newtheorem{proposition}{Proposition}

\graphicspath{%
  {../outputs/phase7/figures/}%
  {../outputs/phase6/figures/}%
  {../outputs/phase5/figures/}%
  {../outputs/phase4/figures/}%
  {../outputs/phase3/figures/pelletizer-I/}%
  {../outputs/phase2/figures/pelletizer-I/}%
}

\title{A Physics-Gated, Label-Free Framework for Industrial Standby
Energy Elimination: State Inference, Duration Forecasting and
Optimisation-Based Shutdown Decisions}

% ---------------------------------------------------------------------
%  AUTHOR BLOCK.  Every FILL_IN below must be replaced before
%  submission.  `generate_paper_draft.py` counts them and prints the
%  count on every run; it does not refuse to assemble, because an
%  unfilled author list is an incomplete submission rather than an
%  inconsistent document.
% ---------------------------------------------------------------------
\author{%
  FILL\_IN Author Name$^{1}$\thanks{Corresponding author:
  \texttt{FILL\_IN author@institution.edu}}\\[2pt]
  \normalsize $^{1}$FILL\_IN Department, FILL\_IN Institution,
  FILL\_IN City, FILL\_IN Country%
}
\date{}

\begin{document}
\maketitle
"""

POSTAMBLE = r"""
% ---------------------------------------------------------------------
%  BACK MATTER.  Placed before the bibliography, which is where every
%  target venue puts it.  The data-availability statement is factual as
%  written except for the FILL_IN items: the code repository and the
%  IMDELD DOI are checked, the second-site records are not published and
%  their provenance has to be stated by the author.
% ---------------------------------------------------------------------
\section*{Acknowledgements}

FILL\_IN: funding source and grant number, or the sentence ``This
research received no specific grant from any funding agency in the
public, commercial, or not-for-profit sectors.'' if unfunded.

\section*{Data Availability}

The IMDELD dataset analysed in Sections~\ref{sec:states}%
--\ref{sec:crossmachine} is publicly available from IEEE DataPort at
\url{https://doi.org/10.21227/cg5v-dk02}~\cite{imdeld2018dataset}. All
experiment code, the stored per-phase outputs from which every table and
figure in this paper is generated, and the assembly script that builds
this document are available at
\url{https://github.com/Ananthvarshan/spark-energy-prediction}. The two
single-channel second-site records used in
Section~\ref{sec:crossmachine:dataset} are FILL\_IN: state whether these
are available on request, under what licence, and from whom.

\section*{Declaration of Competing Interest}

FILL\_IN: ``The authors declare that they have no known competing
financial interests or personal relationships that could have appeared
to influence the work reported in this paper.'' -- or the disclosure.

% `plain` is chosen because it ships with every TeX distribution and
% therefore always builds.  For submission, swap in the venue's style --
% \bibliographystyle{IEEEtran} or \bibliographystyle{elsarticle-num} --
% which is the only change the bibliography needs.
\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:eq|c|C)?ref\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
BIBKEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\bXX\b|\bTODO\b|\bFIXME\b")

# Author name, affiliation, funding, competing-interest and the
# second-site data provenance cannot be filled in from the repository.
# They are marked rather than guessed, counted rather than ignored, and
# deliberately NOT treated as a check failure: an unfilled author list is
# an incomplete submission, not an inconsistent document, and blocking
# assembly on it would stop the draft being read.
FILLIN_RE = re.compile(r"FILL\\?_IN")

# The generated front and back matter is checked like any section file:
# it carries \ref and \cite, and a dangling one there breaks the build
# exactly as it would in the body.
FRONTBACK = "main.tex front/back matter"


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, respecting \\% escapes."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        cleaned, i = [], 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                cleaned.append(line[i:i + 2])
                i += 2
                continue
            if ch == "%":
                break
            cleaned.append(ch)
            i += 1
        out.append("".join(cleaned))
    return "\n".join(out)


def read_body(stem: str) -> tuple[str, str]:
    """Return (raw, comment-stripped) text of a paper source file."""
    path = PAPER / f"{stem}.tex" if not stem.startswith("..") else ROOT / "paper" / f"{stem}.tex"
    if stem.startswith(".."):
        path = (PAPER / f"{stem}.tex").resolve()
    raw = path.read_text(encoding="utf-8")
    return raw, strip_comments(raw)


BEGIN_RE = re.compile(r"\\begin\{([^}]*)\}")
END_RE = re.compile(r"\\end\{([^}]*)\}")
TABULAR_RE = re.compile(
    r"\\begin\{(tabular\*?|tabularx)\}(?:\{[^{}]*\})?\s*\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)
MULTICOL_RE = re.compile(r"\\multicolumn\{(\d+)\}")
RULE_ONLY_RE = re.compile(
    r"^\s*(?:\\(?:top|mid|bottom)rule|\\hline|\\addlinespace(?:\[[^\]]*\])?"
    r"|\\cmidrule(?:\([^)]*\))?(?:\{[^}]*\})?|\s)*$"
)


def count_columns(spec: str) -> int:
    """Number of columns a tabular preamble declares."""
    n, i = 0, 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lcrX":
            n += 1
            i += 1
        elif ch in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            n += 1
            i = skip_group(spec, i + 1)
        elif ch in "@!><":
            i = skip_group(spec, i + 1) if i + 1 < len(spec) and spec[i + 1] == "{" else i + 1
        elif ch == "*" and i + 1 < len(spec) and spec[i + 1] == "{":
            # *{n}{spec} -- expand
            close = skip_group(spec, i + 1)
            reps = int(spec[i + 2:close - 1])
            inner_close = skip_group(spec, close)
            n += reps * count_columns(spec[close + 1:inner_close - 1])
            i = inner_close
        else:
            i += 1
    return n


def skip_group(s: str, open_idx: int) -> int:
    """Index just past the '}' matching the '{' at open_idx."""
    depth, i = 0, open_idx
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def split_fields(row: str) -> int:
    """Count '&'-separated fields at brace depth zero, honouring \\multicolumn."""
    depth, fields, i = 0, 1, 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0:
            fields += 1
        i += 1
    for m in MULTICOL_RE.findall(row):
        fields += int(m) - 1
    return fields


def check_structure(stem: str, body: str) -> list[str]:
    """Environment balance and tabular column counts."""
    problems: list[str] = []

    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        elif not stack:
            problems.append(f"[env]   {stem}.tex: \\end{{{env}}} with no matching \\begin")
        elif stack[-1] != env:
            problems.append(f"[env]   {stem}.tex: \\end{{{env}}} closes \\begin{{{stack[-1]}}}")
            stack.pop()
        else:
            stack.pop()
    for env in stack:
        problems.append(f"[env]   {stem}.tex: \\begin{{{env}}} never closed")

    for m in TABULAR_RE.finditer(body):
        spec, content = m.group(2), m.group(3)
        want = count_columns(spec)
        if want == 0:
            continue
        line_no = body[: m.start()].count("\n") + 1
        for row in content.split(r"\\"):
            if RULE_ONLY_RE.match(row) or not row.strip():
                continue
            got = split_fields(row)
            if got != want:
                snippet = " ".join(row.split())[:52]
                problems.append(
                    f"[table] {stem}.tex:~{line_no}: row has {got} fields, "
                    f"spec declares {want}: {snippet}"
                )
    return problems


def find_graphic(name: str) -> Path | None:
    for base in GRAPHICS_PATHS:
        for ext in GRAPHICS_EXTS:
            candidate = base / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None


# ---------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------

def collect() -> dict:
    """Read every included file and index its labels, refs, cites, figures."""
    stems = [s for s, _ in SECTIONS] + list(NESTED) + [ABLATION_TABLE]
    bodies = {}
    for stem in stems:
        _, body = read_body(stem)
        bodies[stem] = body
    # PREAMBLE and POSTAMBLE are concatenated so \begin{document} and
    # \end{document} balance; apart they would each report as unclosed.
    bodies[FRONTBACK] = strip_comments(PREAMBLE + POSTAMBLE)
    return bodies


def run_checks(bodies: dict) -> list[str]:
    problems: list[str] = []

    labels: dict[str, list[str]] = {}
    refs: dict[str, set[str]] = {}
    cites: dict[str, set[str]] = {}
    figures: dict[str, set[str]] = {}

    for stem, body in bodies.items():
        for m in LABEL_RE.findall(body):
            labels.setdefault(m, []).append(stem)
        for m in REF_RE.findall(body):
            refs.setdefault(m, set()).add(stem)
        for m in CITE_RE.findall(body):
            for key in m.split(","):
                cites.setdefault(key.strip(), set()).add(stem)
        for m in GRAPHIC_RE.findall(body):
            figures.setdefault(m.strip(), set()).add(stem)

    # 1. cross-references
    for key, where in sorted(labels.items()):
        if len(where) > 1:
            problems.append(f"[label] '{key}' defined {len(where)}x: {', '.join(where)}")
    for key, where in sorted(refs.items()):
        if key not in labels:
            problems.append(f"[ref]   '{key}' has no \\label  (cited in {', '.join(sorted(where))})")

    # 2. citations
    bib_keys = set(BIBKEY_RE.findall(BIB.read_text(encoding="utf-8")))
    for key, where in sorted(cites.items()):
        if key not in bib_keys:
            problems.append(f"[cite]  '{key}' not in references.bib  (in {', '.join(sorted(where))})")

    # 3. figures
    for name, where in sorted(figures.items()):
        if find_graphic(name) is None:
            problems.append(f"[fig]   '{name}' not found on any graphics path  (in {', '.join(sorted(where))})")

    # 4. placeholders
    for stem, body in sorted(bodies.items()):
        for n, line in enumerate(body.split("\n"), 1):
            if PLACEHOLDER_RE.search(line):
                problems.append(f"[todo]  {stem}.tex:{n}: {line.strip()[:70]}")

    # 5. orphaned floats.  A table or figure the body never refers to is
    #    one a reader has no reason to look at, and in a two-column
    #    layout it will drift somewhere unhelpful.  Section labels are
    #    exempt: they exist to be cross-referenced from anywhere,
    #    including from files outside this document.
    for key, where in sorted(labels.items()):
        if key.split(":")[0] in ("tab", "fig") and key not in refs:
            problems.append(f"[float] '{key}' is never \\ref'd  (defined in {where[0]})")

    # 6. structure -- what pdflatex would reject, checked without pdflatex
    for stem, body in sorted(bodies.items()):
        problems.extend(check_structure(stem, body))

    return problems


def report(bodies: dict, problems: list[str]) -> None:
    labels = sum(len(LABEL_RE.findall(b)) for b in bodies.values())
    refs = sum(len(REF_RE.findall(b)) for b in bodies.values())
    cites = sum(len(CITE_RE.findall(b)) for b in bodies.values())
    figs = sum(len(GRAPHIC_RE.findall(b)) for b in bodies.values())
    skip = {ABLATION_TABLE, FRONTBACK}
    words = sum(len(b.split()) for s, b in bodies.items() if s not in skip)
    fillins = sum(len(FILLIN_RE.findall(b)) for b in bodies.values())

    print(f"  files      {len(bodies)}")
    print(f"  ~words     {words:,}   (body text, comments stripped)")
    print(f"  labels     {labels}")
    print(f"  refs       {refs}")
    print(f"  citations  {cites}")
    print(f"  figures    {figs}")
    print()

    # Reported, not counted as a problem.  See FILLIN_RE.
    if fillins:
        print(f"  {fillins} FILL_IN marker(s) outstanding -- author name and")
        print("  affiliation, funding, competing-interest declaration, and the")
        print("  second-site data provenance.  These cannot be filled from the")
        print("  repository and must be supplied before submission:")
        for stem, body in sorted(bodies.items()):
            for n, line in enumerate(body.split("\n"), 1):
                if FILLIN_RE.search(line):
                    print(f"    {stem}:{n}: {line.strip()[:64]}")
        print()
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    {p}")
    else:
        print("  all six checks pass: refs and cites resolve, figures exist,")
        print("  no placeholders, no orphaned floats, environments and tables")
        print("  are well formed.  Nothing here proves it builds -- there is no")
        print("  TeX distribution on this machine -- but nothing statically")
        print("  detectable is wrong with it.")
    print()


def assemble() -> str:
    parts = [PREAMBLE, "\n\\input{abstract}\n"]
    for stem, description in SECTIONS:
        parts.append(f"\n% --- {description}\n\\input{{{stem}}}\n")
        if stem == "ablation":
            parts.append(f"\\input{{{ABLATION_TABLE}}}\n")
    parts.append(POSTAMBLE)
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="check only, do not write")
    ap.add_argument("--force", action="store_true", help="write even if checks fail")
    args = ap.parse_args()

    print("=" * 62)
    print("PAPER DRAFT ASSEMBLY")
    print("=" * 62)

    bodies = collect()
    # the abstract is included by main.tex but is not a numbered section
    bodies["abstract"] = read_body("abstract")[1]

    problems = run_checks(bodies)
    report(bodies, problems)

    if args.check:
        return 1 if problems else 0

    if problems and not args.force:
        print("  NOT WRITING main.tex.  Fix the problems above, or pass --force.")
        return 1

    OUT.write_text(assemble(), encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)}"
          f"{'  (forced, with problems)' if problems else ''}")
    print()
    print("  to build:  cd paper && pdflatex main && bibtex main && "
          "pdflatex main && pdflatex main")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
