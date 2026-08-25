"""Per-run audit manifest: what was read, what was done, with what code.

A processed channel is a claim about a physical measurement. Six months later
somebody has to be able to establish which file produced it, which parameters
were used, which version of this code ran, and whether the source file has since
changed. Without that, a number in a report is unfalsifiable.

Everything here is recorded, never inferred. If a fact is unavailable (an x0
that the file format does not carry, a git commit in a non-git checkout) the
manifest says so explicitly rather than filling in a plausible default -- an
unverified value that looks verified is worse than a gap.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy

__all__ = ["file_checksum", "git_commit", "library_versions", "X0Provenance",
           "RunManifest"]

_CHUNK = 1 << 20


def file_checksum(path: Path, algo: str = "sha256") -> str:
    """Streaming checksum, so a 3 GB raw file does not have to be resident."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            h.update(block)
    return f"{algo}:{h.hexdigest()}"


def git_commit(repo_root: Path) -> str:
    """Current commit, or an explicit marker. Never a guess.

    A dirty tree is reported as such: the commit alone would misrepresent what
    actually ran, since uncommitted edits are exactly the changes most likely to
    be the reason a result looks different.
    """
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unavailable: not a git checkout"
        commit = rev.stdout.strip()
        st = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
        dirty = bool(st.stdout.strip())
        return f"{commit}{'+dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def library_versions() -> Dict[str, str]:
    """Versions of everything that can change a floating-point result."""
    import famos
    return {
        "famos": famos.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@dataclass
class X0Provenance:
    """Where the record's start time came from -- or that it did not.

    `verified=False` means the value was supplied, not read. It propagates into
    the manifest and marks every derived time value as unverified, because an
    assumed x0 shifts the entire time axis and nothing downstream can detect it.
    """
    value_s: float
    verified: bool
    source: str                     # e.g. "IMC2 CD block, field 8, offset 0x81"
    marker: Optional[str] = None
    field_index: Optional[int] = None
    byte_offset: Optional[int] = None
    note: str = ""

    @classmethod
    def assumed(cls, value_s: float, why: str) -> "X0Provenance":
        return cls(value_s=value_s, verified=False, source="ASSUMED", note=why)


@dataclass
class RunManifest:
    """The full record of one processing run."""
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    chain: List[Dict[str, Any]] = field(default_factory=list)
    x0: Optional[Dict[str, Any]] = None
    libraries: Dict[str, str] = field(default_factory=library_versions)
    git: str = ""
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_input(self, path: Path, **extra: Any) -> None:
        p = Path(path)
        self.inputs.append({
            "path": str(p), "bytes": p.stat().st_size,
            "checksum": file_checksum(p), **extra})

    def add_stage(self, famos_function: str, **params: Any) -> None:
        """Record a stage by the FAMOS function it reproduces, plus parameters."""
        self.chain.append({"famos_function": famos_function, **params})

    def set_x0(self, prov: X0Provenance) -> None:
        self.x0 = asdict(prov)
        if not prov.verified:
            self.warnings.append(
                f"x0 = {prov.value_s} was ASSUMED, not read from the file "
                f"({prov.note}). Every time value in this run is unverified.")

    def add_output(self, path: Path, n_samples: int, fs_hz: float,
                   **extra: Any) -> None:
        p = Path(path)
        self.outputs.append({
            "path": str(p), "samples": int(n_samples), "fs_hz": float(fs_hz),
            "checksum": file_checksum(p) if p.exists() else "not written",
            **extra})

    def finalise(self, repo_root: Path) -> "RunManifest":
        self.git = git_commit(repo_root)
        return self

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [f"run          : {self.started_utc}",
                 f"git          : {self.git}",
                 f"famos/np/sp  : {self.libraries.get('famos')} / "
                 f"{self.libraries.get('numpy')} / {self.libraries.get('scipy')}"]
        for i in self.inputs:
            lines.append(f"input        : {Path(i['path']).name}  {i['checksum'][:23]}...")
        for c in self.chain:
            ps = ", ".join(f"{k}={v}" for k, v in c.items() if k != "famos_function")
            lines.append(f"stage        : {c['famos_function']}({ps})")
        if self.x0:
            flag = "verified" if self.x0["verified"] else "ASSUMED - UNVERIFIED"
            lines.append(f"x0           : {self.x0['value_s']} s [{flag}] "
                         f"from {self.x0['source']}")
        for w in self.warnings:
            lines.append(f"WARNING      : {w}")
        return "\n".join(lines)
