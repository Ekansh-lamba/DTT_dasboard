"""Dependency licence report, and a build gate that fails on a bad licence.

The constraint is unlimited commercial use with no per-seat cost, so this is a
release blocker, not a formality. Implemented on `importlib.metadata` from the
standard library rather than pip-licenses, because adding a dependency to police
dependencies is a poor trade and pip-licenses is not always installed in CI.

Policy
------
ALLOWED   BSD, MIT, Apache-2.0, PSF, ISC, MPL-2.0 and their common spellings.
REVIEW    LGPL in any form -- usable in most corporate settings but it is a
          legal decision, not an engineering one, so the gate stops and asks.
REJECTED  GPL, AGPL, SSPL, and anything with a non-commercial or field-of-use
          restriction.
UNKNOWN   Treated as REVIEW. A package whose licence cannot be determined is
          not thereby safe; it is simply unexamined.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, List, Optional, Tuple

__all__ = ["classify", "report", "gate", "ALLOWED", "REVIEW", "REJECTED"]

ALLOWED = "ALLOWED"
REVIEW = "REVIEW"
REJECTED = "REJECTED"

# Order matters: the rejecting patterns are tested before the permissive ones,
# so "GNU Lesser General Public License" cannot match a bare "GPL" rule first,
# and "AGPL" cannot be read as "GPL".
_REJECT = [
    (re.compile(r"\bAGPL|AFFERO\b", re.I), "AGPL"),
    (re.compile(r"\bSSPL|SERVER SIDE PUBLIC\b", re.I), "SSPL"),
    (re.compile(r"NON[- ]?COMMERCIAL|NONCOMMERCIAL", re.I), "non-commercial clause"),
    (re.compile(r"\bCC[- ]BY[- ]NC", re.I), "Creative Commons NonCommercial"),
]
_REVIEW = [
    (re.compile(r"\bLGPL\b|LESSER GENERAL PUBLIC", re.I), "LGPL"),
]
_REJECT_GPL = re.compile(r"\bGPL\b|GENERAL PUBLIC LICEN[SC]E", re.I)
_ALLOW = re.compile(
    r"\bBSD\b|\bMIT\b|APACHE|\bPSF\b|PYTHON SOFTWARE FOUNDATION|\bISC\b|"
    r"\bMPL\b|MOZILLA PUBLIC|HISTORICAL PERMISSION|\bZLIB\b|UNLICENSE|"
    r"PUBLIC DOMAIN|\bCC0\b", re.I)
# Many wheels ship the full licence TEXT rather than an SPDX identifier, so
# the body has to be recognised too. Without this, permissive packages
# (scipy, matplotlib, cycler, kiwisolver) land in REVIEW and the gate cries
# wolf -- worse than useless, because a noisy gate gets waived wholesale and
# the one real LGPL finding goes through with it.
_ALLOW_BODY = re.compile(
    r"REDISTRIBUTION AND USE IN SOURCE AND BINARY FORMS"        # BSD family
    r"|PERMISSION IS HEREBY GRANTED, FREE OF CHARGE"            # MIT
    r"|LICENSE AGREEMENT FOR MATPLOTLIB"                        # PSF-derived
    r"|PERMISSION TO USE, COPY, MODIFY, AND(/OR)? DISTRIBUTE",  # ISC
    re.I)


@dataclass
class PackageLicence:
    name: str
    version: str
    licence: str
    verdict: str
    reason: str = ""


def classify(licence_text: str) -> Tuple[str, str]:
    """Classify a licence string into ALLOWED / REVIEW / REJECTED."""
    if not licence_text or licence_text.strip().lower() in ("", "unknown", "none"):
        return REVIEW, "licence could not be determined"
    for pat, label in _REJECT:
        if pat.search(licence_text):
            return REJECTED, label
    for pat, label in _REVIEW:
        if pat.search(licence_text):
            return REVIEW, label
    if _REJECT_GPL.search(licence_text):
        return REJECTED, "GPL"
    if _ALLOW.search(licence_text) or _ALLOW_BODY.search(licence_text):
        return ALLOWED, ""
    return REVIEW, "unrecognised licence string"


def _licence_of(dist: metadata.Distribution) -> str:
    """Best available licence string for a distribution.

    Modern wheels often leave `License:` empty and express the licence only via
    trove classifiers, so both are consulted. Newer metadata may also use the
    PEP 639 `License-Expression` field.
    """
    md = dist.metadata
    for key in ("License-Expression", "License"):
        val = md.get(key)
        if val and val.strip().lower() not in ("unknown", "none"):
            # Keep enough of a full licence body to recognise it,
            # but not so much that the report becomes unreadable.
            return " ".join(val.strip().split())[:400]
    classifiers = md.get_all("Classifier") or []
    lic = [c.split("::")[-1].strip() for c in classifiers
           if c.startswith("License ::")]
    return "; ".join(lic) if lic else ""


def report(only: Optional[List[str]] = None) -> List[PackageLicence]:
    """Licence verdict for every installed distribution (or a named subset)."""
    out: List[PackageLicence] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or "?"
        if only and name.lower() not in {o.lower() for o in only}:
            continue
        lic = _licence_of(dist)
        verdict, reason = classify(lic)
        out.append(PackageLicence(name=name, version=dist.version or "?",
                                  licence=lic or "(none declared)",
                                  verdict=verdict, reason=reason))
    return sorted(out, key=lambda p: (p.verdict != REJECTED,
                                      p.verdict != REVIEW, p.name.lower()))


def gate(only: Optional[List[str]] = None, allow_review: bool = False) -> int:
    """Print the report and return a process exit code.

    Non-zero on any REJECTED package, and on REVIEW unless explicitly waived --
    an unreviewed LGPL or unknown licence is not a pass, it is a pending
    decision, and letting the build go green would bury it.
    """
    rows = report(only)
    width = max((len(r.name) for r in rows), default=10)
    print(f"{'PACKAGE'.ljust(width)}  {'VERSION':<12} {'VERDICT':<9} LICENCE")
    print("-" * (width + 45))
    for r in rows:
        note = f"  <- {r.reason}" if r.reason else ""
        print(f"{r.name.ljust(width)}  {r.version:<12} {r.verdict:<9} "
              f"{r.licence[:44]}{note}")

    bad = [r for r in rows if r.verdict == REJECTED]
    review = [r for r in rows if r.verdict == REVIEW]
    print()
    print(f"  allowed  : {sum(1 for r in rows if r.verdict == ALLOWED)}")
    print(f"  review   : {len(review)}")
    print(f"  rejected : {len(bad)}")

    if bad:
        print("\nFAIL - licences outside the permitted set:")
        for r in bad:
            print(f"  {r.name} {r.version}: {r.licence} ({r.reason})")
        return 2
    if review and not allow_review:
        print("\nFAIL - licences needing a legal decision before release:")
        for r in review:
            print(f"  {r.name} {r.version}: {r.licence} ({r.reason})")
        print("\nResolve each, or re-run with --allow-review once a decision "
              "is on record.")
        return 1
    print("\nPASS - every dependency is free for unlimited commercial use.")
    return 0


if __name__ == "__main__":
    sys.exit(gate())
