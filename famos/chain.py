"""Declarative chains: a YAML file that says exactly which FAMOS operators run.

A chain is data, not code, so the processing applied to a study is reviewable by
someone who does not read Python, diffable in version control, and attachable to
a report. `from_seq` translates an existing imc sequence into one, flagging
anything it cannot map rather than dropping it.

Example
-------
    name: wft_road_load
    sample_rate_hz: 1000.0
    channels:
      Latacc:
        - {op: filtlp, order: 4, cutoff_hz: 5.0}   # FiltLP(Lat_acc,0,0,4,5)
        - {op: smo, width_s: 0.5}                  # smo(Latacc_LPF,0.5)
      "FL_F*":
        - {op: smo, width_s: 0.1}                  # smo(FL_Fx1,0.1)
    finally:
      - {op: red, factor: 10}                      # red(cutd,10) -- applies to all
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from famos import ops
from famos.audit import RunManifest

__all__ = ["Stage", "Chain", "load_chain", "from_seq", "SeqTranslation"]

# Operators this library implements. Anything else in a chain is an error, not
# a no-op: a stage that silently does nothing produces output that looks
# processed but is not.
_OPS = {"filtlp", "smo", "red"}


@dataclass(frozen=True)
class Stage:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(
                f"unknown operator {self.op!r}; implemented: {sorted(_OPS)}. "
                "Refusing to skip it -- an unimplemented stage must fail, not "
                "quietly pass the signal through.")

    def apply(self, x: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, float]:
        """Run this stage, returning ``(signal, new_fs)``."""
        p = self.params
        if self.op == "filtlp":
            return ops.filtlp(x, p["cutoff_hz"], p["order"], fs_hz,
                              init=p.get("init", ops.INIT_FAMOS),
                              character=p.get("character", ops.CHARACTER_BUTTERWORTH),
                              parameter=p.get("parameter", 0.0)), fs_hz
        if self.op == "smo":
            return ops.smo(x, p["width_s"], fs_hz), fs_hz
        if self.op == "red":
            f = int(p["factor"])
            return ops.red(x, f), fs_hz / f
        raise AssertionError(f"unreachable: {self.op}")

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.op}({ps})"


@dataclass
class Chain:
    name: str
    sample_rate_hz: Optional[float] = None
    channels: Dict[str, List[Stage]] = field(default_factory=dict)
    final: List[Stage] = field(default_factory=list)

    def stages_for(self, channel: str) -> List[Stage]:
        """Stages for one channel: the first matching pattern, then `finally`.

        Exact names win over globs, and among globs the most specific (longest
        pattern) wins, so `FL_Fx` beats `FL_F*` beats `*`. Without a defined
        precedence, which rule applied would depend on dict ordering -- and a
        channel silently taking the wrong filter is the worst kind of bug here,
        because the output is still a plausible signal.
        """
        if channel in self.channels:
            return list(self.channels[channel]) + list(self.final)
        globs = [p for p in self.channels
                 if not p.isidentifier() and fnmatch.fnmatch(channel, p)]
        if globs:
            best = max(globs, key=len)
            return list(self.channels[best]) + list(self.final)
        return list(self.final)

    def run(self, channel: str, x, fs_hz: Optional[float] = None,
            manifest: Optional[RunManifest] = None) -> Tuple[np.ndarray, float]:
        fs = fs_hz if fs_hz is not None else self.sample_rate_hz
        if fs is None:
            raise ValueError(
                "sample rate not given and not declared in the chain. It must "
                "come from the file header -- never a default.")
        y = ops.as_f64(x, f"{channel} input")
        for st in self.stages_for(channel):
            y, fs = st.apply(y, fs)
            if manifest is not None:
                manifest.add_stage(_FAMOS_NAME[st.op], channel=channel, **st.params)
        return y, fs


_FAMOS_NAME = {"filtlp": "FiltLP", "smo": "Smo", "red": "red"}


def _stage(d: Dict[str, Any]) -> Stage:
    d = dict(d)
    op = d.pop("op", None)
    if op is None:
        raise ValueError(f"stage is missing its 'op' key: {d}")
    return Stage(op=str(op), params=d)


def load_chain(path: Path) -> Chain:
    """Read a chain from YAML. Malformed input raises; nothing is defaulted."""
    import yaml

    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    if "name" not in doc:
        raise ValueError(f"{path}: 'name' is required")
    chans = {k: [_stage(s) for s in v]
             for k, v in (doc.get("channels") or {}).items()}
    return Chain(
        name=str(doc["name"]),
        sample_rate_hz=(float(doc["sample_rate_hz"])
                        if doc.get("sample_rate_hz") is not None else None),
        channels=chans,
        final=[_stage(s) for s in (doc.get("finally") or [])])


# ------------------------------------------------------- .seq -> chain

@dataclass
class SeqTranslation:
    chain: Chain
    unmapped: List[str] = field(default_factory=list)
    renames: Dict[str, str] = field(default_factory=dict)

    def report(self) -> str:
        out = [f"translated chain: {self.chain.name}",
               f"  channels with stages : {len(self.chain.channels)}",
               f"  final stages         : {len(self.chain.final)}",
               f"  renames parsed       : {len(self.renames)}"]
        if self.unmapped:
            out.append(f"  NOT TRANSLATED ({len(self.unmapped)}) "
                       "-- review each before trusting this chain:")
            out += [f"    {u}" for u in self.unmapped]
        else:
            out.append("  every statement was translated")
        return "\n".join(out)


_RE_ASSIGN = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.+?)\s*$")
_RE_FILTLP = re.compile(
    r"^FiltLP\(\s*([A-Za-z_][\w.]*)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,"
    r"\s*(\d+)\s*,\s*([-\d.eE+]+)\s*\)$", re.I)
_RE_SMO = re.compile(r"^smo\(\s*([A-Za-z_][\w.]*)\s*,\s*([-\d.eE+]+)\s*\)$", re.I)
_RE_RED = re.compile(r"^red\(\s*([A-Za-z_][\w.]*)\s*,\s*(\d+)\s*\)$", re.I)
_RE_NAME = re.compile(r"^[A-Za-z_][\w.]*$")


def from_seq(path: Path, name: Optional[str] = None) -> SeqTranslation:
    """Translate an imc sequence into a Chain, flagging what it cannot map.

    Resolves operator chains through intermediates: the recipe writes
    ``Latacc_LPF = FiltLP(Lat_acc,...)`` then ``Latacc = smo(Latacc_LPF,0.5)``,
    which is one two-stage chain on ``Lat_acc``, not two separate channels.

    Everything it cannot map is listed in `unmapped`. Nothing is dropped
    silently -- an untranslated stage means the YAML is not equivalent to the
    sequence, and the operator has to know that before trusting it.
    """
    path = Path(path)
    text = path.read_text(encoding="latin-1")

    # name -> (root raw channel, stages applied to it so far), built in FILE
    # ORDER so that in-place redefinition works. FAMOS routinely writes
    # `Long_acc = smo(Long_acc, 0.5)`, i.e. a channel updated in place. Reading
    # the old binding before writing the new one is what makes that resolve;
    # treating it as a self-reference instead sends a naive resolver into a
    # cycle and silently drops the stage.
    defs: Dict[str, Tuple[str, List[Stage]]] = {}
    consumed: set = set()          # names used as another name's source
    renames: Dict[str, str] = {}
    final: List[Stage] = []
    unmapped: List[str] = []

    def base_of(src: str) -> Tuple[str, List[Stage]]:
        root, stages = defs.get(src, (src, []))
        return root, list(stages)

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.upper().startswith("YUNIT"):
            continue
        m = _RE_ASSIGN.match(line)
        if not m:
            continue
        target, expr = m.group(1), m.group(2)

        if mm := _RE_FILTLP.match(expr):
            src = mm.group(1)
            root, stages = base_of(src)
            if src != target:
                consumed.add(src)
            defs[target] = (root, stages + [Stage("filtlp", {
                "order": int(mm.group(4)), "cutoff_hz": float(mm.group(5)),
                "character": int(float(mm.group(2))),
                "parameter": float(mm.group(3))})])
        elif mm := _RE_SMO.match(expr):
            src = mm.group(1)
            root, stages = base_of(src)
            if src != target:
                consumed.add(src)
            defs[target] = (root,
                            stages + [Stage("smo", {"width_s": float(mm.group(2))})])
        elif mm := _RE_RED.match(expr):
            # red() is applied to a group holding every channel, so it is a
            # chain-wide final stage rather than one channel's stage.
            final.append(Stage("red", {"factor": int(mm.group(2))}))
        elif _RE_NAME.match(expr):
            if expr != target:
                renames[target] = expr
                consumed.add(expr)
                root, stages = base_of(expr)
                defs[target] = (root, stages)
            # `X = X` is a no-op restatement; it neither renames nor consumes.
        else:
            unmapped.append(f"{path.name}:{lineno}: {line}")

    # Keep only terminal names -- those nothing else consumed. An intermediate
    # such as Latacc_LPF is folded into the chain of the channel that used it.
    channels: Dict[str, List[Stage]] = {}
    for defined_name, (root, stages) in defs.items():
        # NOT `name`: that is the function parameter holding the chain's name,
        # and shadowing it here left the chain named after whichever channel
        # happened to be last in the dict.
        if defined_name in consumed or not stages:
            continue
        prev = channels.get(root)
        if prev is None or len(stages) > len(prev):
            channels[root] = stages

    return SeqTranslation(
        chain=Chain(name=name or path.stem, channels=channels, final=final),
        unmapped=unmapped, renames=renames)
