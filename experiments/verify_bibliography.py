"""
============================================================
BIBLIOGRAPHY VERIFICATION  --  experiments/verify_bibliography.py
============================================================

PURPOSE
-------
`paper/references.bib` was written over seven phases, and a subset of
its volume/issue/page/DOI fields were reconstructed from domain
knowledge rather than copied from a publisher record.  Sixteen entries
carry a `% VERIFY` comment saying so.  An incorrect DOI is a
referee-visible error that has nothing to do with the science, and it
is entirely mechanical to catch: resolve every DOI against Crossref and
compare the fields.

    python -m experiments.verify_bibliography            # check all
    python -m experiments.verify_bibliography --flagged  # only % VERIFY
    python -m experiments.verify_bibliography --json out.json

WHAT IS COMPARED, AND HOW STRICTLY
-----------------------------------
Crossref is authoritative for DOI, container title, volume, issue,
pages and year; it is NOT reliable for author name spelling (accents
and initials are inconsistently deposited), so authors are compared on
surname sets only and reported as a note rather than a mismatch.

Titles are compared after case-folding, stripping punctuation and
removing LaTeX braces, because `{M}arkov` and `Markov` are the same
title and a naive comparison would report every protected-capital entry
as wrong.

Page ranges are normalised: Crossref deposits `215-243`, BibTeX wants
`215--243`, and some publishers deposit only a first page or an article
number.  A BibTeX range whose first page matches a Crossref single page
is reported as a note, not a mismatch.

EXIT STATUS
-----------
0 if every resolvable DOI matched on the strict fields, 1 otherwise.
An unresolvable DOI is always a failure: a DOI that does not resolve is
worse than no DOI at all, because a reader will try it.

Entries with no DOI (conference proceedings, standards, books) cannot
be checked here and are listed as such, so the residue needing manual
attention is explicit rather than assumed empty.
============================================================
"""

from __future__ import annotations

import re
import sys
import json
import time
import argparse
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "references.bib"

CROSSREF = "https://api.crossref.org/works/"
DATACITE = "https://api.datacite.org/dois/"
# Crossref asks for a contact address in the User-Agent; it buys the
# "polite" request pool and is the documented way to use the API.
UA = ("bib-verify/1.0 (https://github.com/Ananthvarshan/"
      "spark-energy-prediction; mailto:hydromoorthy303@gmail.com)")

ENTRY_RE = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.DOTALL)
FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\n?$)",
                      re.DOTALL)


def parse_bib(text: str) -> list[dict]:
    """Parse enough BibTeX to compare fields.  Not a general parser."""
    out = []
    for m in ENTRY_RE.finditer(text):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in FIELD_RE.finditer(body):
            fields[fm.group(1).lower()] = " ".join(fm.group(2).split())
        tail = text[m.end():m.end() + 80].split("\n")[0]
        out.append({"type": kind, "key": key, "fields": fields,
                    "flagged": "VERIFY" in tail})
    return out


def norm_title(s: str) -> str:
    # Crossref titles carry JATS markup -- <i>EM</i>, <sub>2</sub>,
    # <scp>Markov</scp> -- which must go before punctuation is squashed,
    # or the tag letters survive as words ("i em i algorithm").
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)          # \'e and friends
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def norm_pages(s: str) -> str:
    s = s.replace("--", "-").replace("–", "-")
    return " ".join(s.split()).strip()


def surnames(bib_author: str) -> set[str]:
    """Surnames from a BibTeX author list, lowercased and de-accented."""
    out = set()
    for a in re.split(r"\s+and\s+", bib_author):
        a = re.sub(r"[{}\\'`\"^~]", "", a).strip()
        if not a:
            continue
        surname = a.split(",")[0] if "," in a else a.split()[-1]
        out.add(surname.lower().strip())
    return out


def _datacite(doi: str) -> dict | None:
    """
    Fall back to DataCite for DOIs Crossref does not hold.

    Datasets and repository deposits -- IEEE DataPort, Zenodo, Figshare
    -- register with DataCite, not Crossref, so a perfectly valid
    dataset DOI 404s against the Crossref API.  Reporting those as
    broken would be wrong, and dropping the check would leave the one
    DOI a reader is most likely to follow unverified.  The record is
    reshaped into the few Crossref-style keys `compare` reads.
    """
    req = urllib.request.Request(DATACITE + urllib.parse.quote(doi),
                                 headers={"User-Agent": UA,
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as fh:
            at = json.load(fh)["data"]["attributes"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    year = at.get("publicationYear")
    return {
        "title": [t.get("title", "") for t in (at.get("titles") or [{}])],
        "issued": {"date-parts": [[year]]} if year else {},
        "container-title": [at.get("publisher") or ""],
        "author": [{"family": (c.get("familyName") or
                               (c.get("name") or "").split(",")[0])}
                   for c in (at.get("creators") or [])],
        "_registry": "DataCite",
    }


def fetch(doi: str) -> dict | None:
    req = urllib.request.Request(CROSSREF + urllib.parse.quote(doi),
                                 headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as fh:
            return json.load(fh)["message"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return _datacite(doi)
        raise


def compare(entry: dict, cr: dict) -> tuple[list[str], list[str]]:
    """Return (mismatches, notes) for one entry against its Crossref record."""
    f = entry["fields"]
    bad, notes = [], []

    # ACM and several others deposit the subtitle in its own field, so
    # Crossref's `title` for "XGBoost: A Scalable Tree Boosting System"
    # is the bare word "XGBoost".  Joining the two before comparing
    # stops every such entry reporting as wrong.
    cr_title = (cr.get("title") or [""])[0]
    cr_sub = (cr.get("subtitle") or [""])[0] if cr.get("subtitle") else ""
    cr_full = f"{cr_title}: {cr_sub}" if cr_sub else cr_title
    nb = norm_title(f.get("title", ""))
    if cr_title and nb not in (norm_title(cr_full), norm_title(cr_title)):
        # Some deposits are simply truncated -- Wiley's record for the EM
        # paper ends mid-sentence at "...Incomplete Data Via the".  A
        # Crossref title that is a proper prefix of the bib title is a bad
        # deposit, not a bad citation, so it is a note.
        if nb.startswith(norm_title(cr_title)):
            notes.append(f'title: crossref deposit truncated to '
                         f'"{cr_title.strip()[:52]}"')
        else:
            bad.append(f'title: bib "{f.get("title","")[:58]}" '
                       f'!= crossref "{cr_full[:58]}"')

    cr_year = None
    for k in ("issued", "published-print", "published-online", "published"):
        parts = (cr.get(k) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            cr_year = parts[0][0]
            break
    if cr_year and f.get("year") and str(cr_year) != f["year"]:
        # A print/online split of one year is routine and not an error.
        if abs(int(f["year"]) - cr_year) <= 1:
            notes.append(f'year: bib {f["year"]}, crossref {cr_year} '
                         f'(print/online split)')
        else:
            bad.append(f'year: bib {f["year"]} != crossref {cr_year}')

    for field, crkey in (("volume", "volume"), ("issue", "issue")):
        want = f.get("number" if field == "issue" else field)
        got = cr.get(crkey)
        # An issue range is written `18--19` in BibTeX and `18-19` by
        # Crossref; normalise both before comparing.
        if want and got and norm_pages(str(got)) != norm_pages(str(want)):
            bad.append(f"{field}: bib {want} != crossref {got}")

    if f.get("pages") and cr.get("page"):
        wp, gp = norm_pages(f["pages"]), norm_pages(cr["page"])
        if wp != gp:
            if wp.split("-")[0] == gp.split("-")[0]:
                notes.append(f"pages: bib {wp}, crossref {gp} (same first page)")
            else:
                bad.append(f"pages: bib {wp} != crossref {gp}")

    cr_names = {(a.get("family") or "").lower()
                for a in (cr.get("author") or []) if a.get("family")}
    bib_names = surnames(f.get("author", ""))
    if cr_names and bib_names and cr_names != bib_names:
        missing = cr_names - bib_names
        extra = bib_names - cr_names
        detail = []
        if missing:
            detail.append(f"not in bib: {', '.join(sorted(missing))}")
        if extra:
            detail.append(f"not in crossref: {', '.join(sorted(extra))}")
        notes.append("authors: " + "; ".join(detail))

    container = (cr.get("container-title") or [""])[0]
    bib_container = f.get("journal") or f.get("booktitle") or ""
    if container and bib_container:
        if norm_title(container) != norm_title(bib_container):
            notes.append(f'venue: bib "{bib_container[:44]}" '
                         f'vs crossref "{container[:44]}"')
    return bad, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flagged", action="store_true",
                    help="only entries carrying a %% VERIFY comment")
    ap.add_argument("--json", metavar="PATH", help="write the full report")
    ap.add_argument("--delay", type=float, default=0.12,
                    help="seconds between Crossref requests")
    args = ap.parse_args()

    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    if args.flagged:
        entries = [e for e in entries if e["flagged"]]

    print("=" * 62)
    print("BIBLIOGRAPHY VERIFICATION vs CROSSREF")
    print("=" * 62)
    print(f"  entries considered   {len(entries)}")

    report, n_bad, n_nodoi, n_unresolved = [], 0, 0, 0
    for e in entries:
        doi = e["fields"].get("doi")
        if not doi:
            n_nodoi += 1
            report.append({"key": e["key"], "status": "no-doi",
                           "flagged": e["flagged"]})
            continue
        time.sleep(args.delay)
        try:
            cr = fetch(doi)
        except Exception as exc:                       # network, 5xx, ...
            report.append({"key": e["key"], "status": "error",
                           "doi": doi, "detail": f"{type(exc).__name__}: {exc}"})
            n_unresolved += 1
            continue
        if cr is None:
            report.append({"key": e["key"], "status": "unresolved", "doi": doi})
            n_unresolved += 1
            continue
        bad, notes = compare(e, cr)
        report.append({"key": e["key"], "status": "mismatch" if bad else "ok",
                       "doi": doi, "flagged": e["flagged"],
                       "mismatches": bad, "notes": notes})
        if bad:
            n_bad += 1

    checked = len(entries) - n_nodoi
    print(f"  DOIs checked         {checked}")
    print(f"  matched              {checked - n_bad - n_unresolved}")
    print(f"  MISMATCHED           {n_bad}")
    print(f"  UNRESOLVED           {n_unresolved}")
    print(f"  no DOI (manual)      {n_nodoi}")
    print()

    for r in report:
        if r["status"] in ("mismatch", "unresolved", "error"):
            head = "UNRESOLVED" if r["status"] != "mismatch" else "MISMATCH"
            print(f"  [{head}] {r['key']}   {r.get('doi','')}")
            for m in r.get("mismatches", []):
                print(f"      ! {m}")
            if r.get("detail"):
                print(f"      ! {r['detail']}")
            for n in r.get("notes", []):
                print(f"      . {n}")
            print()

    noteworthy = [r for r in report
                  if r["status"] == "ok" and r.get("notes")]
    if noteworthy:
        print(f"  {len(noteworthy)} entry(ies) matched on the strict fields "
              f"with notes:")
        for r in noteworthy:
            print(f"    {r['key']}")
            for n in r["notes"]:
                print(f"      . {n}")
        print()

    nodoi = [r["key"] for r in report if r["status"] == "no-doi"]
    if nodoi:
        print(f"  {len(nodoi)} entry(ies) carry no DOI and cannot be checked "
              f"here:")
        print("    " + ", ".join(nodoi))
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1),
                                   encoding="utf-8")
        print(f"  wrote {args.json}")

    return 1 if (n_bad or n_unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
