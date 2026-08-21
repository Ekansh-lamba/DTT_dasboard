

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

COMPONENTS: Tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
FORCE_COMPONENTS: Tuple[str, ...] = ("Fx", "Fy", "Fz")

# Legacy 2-axle names ⇄ canonical
_LEGACY_TO_AXLE = {"FL": (1, "L"), "FR": (1, "R"), "RL": (2, "L"), "RR": (2, "R")}
_AXLE_TO_LEGACY = {v: k for k, v in _LEGACY_TO_AXLE.items()}


@dataclass(frozen=True)
class Position:
    axle: int                     # 1 = front-most
    side: str                     # "L" | "R"
    sub: Optional[str] = None     # None | "I" | "O"  (inner/outer for duals)

    @property
    def id(self) -> str:
        return f"A{self.axle}{self.side}{self.sub or ''}"

    @property
    def legacy_label(self) -> Optional[str]:
        """FL/FR/RL/RR for a 2-axle single-wheel position, else None."""
        if self.sub is None:
            return _AXLE_TO_LEGACY.get((self.axle, self.side))
        return None
    @property
    def display(self) -> str:
        """Verbose human label, e.g. 'Axle 3 Left outer'."""
        io = {"I": " inner", "O": " outer"}.get(self.sub or "", "")
        return f"Axle {self.axle} {'Left' if self.side == 'L' else 'Right'}{io}"
    @property
    def label(self) -> str:
        return self.id

    def channel(self, component: str) -> str:
        return f"{self.id}_{component}"

    def __lt__(self, other: "Position") -> bool:
        order = {"L": 0, "R": 1}
        return (self.axle, order[self.side], self.sub or "") < \
               (other.axle, order[other.side], other.sub or "")


# Discovery

# Ordered patterns: first match wins. Each returns (axle, side, sub|None).
_POS_PATTERNS = [
    # explicit canonical / axle-side: A2R, A2RO, 2R, 2RO, R2
    (re.compile(r"(?:^|[_\W])A?(\d)([LR])([IO])?(?:$|[_\W])", re.I),
     lambda m: (int(m.group(1)), m.group(2).upper(), (m.group(3) or "").upper() or None)),
    # legacy fl/fr/rl/rr (2-axle)
    (re.compile(r"(?:^|[_\W])(fl|fr|rl|rr)(?:$|[_\W])", re.I),
     lambda m: (*_LEGACY_TO_AXLE[m.group(1).upper()], None)),
]

_COMP_RE = re.compile(r"(?:^|[_\W])(F[xyz]|M[xyz])(?:$|[_\W])", re.I)

_DERIVED_RE = re.compile(r"(rot|corr|anglespeed|angle|rpm|speed|accel|_vel|_ws\d)", re.I)


def _norm_component(tok: str) -> str:
    return tok[0].upper() + tok[1].lower()


def parse_channel(name: str) -> Optional[Tuple[Position, str]]:
    """Return (Position, component) for a channel/file name, or None."""
    if _DERIVED_RE.search(name):
        return None
    cm = _COMP_RE.search(name)
    if not cm:
        return None
    comp = _norm_component(cm.group(1))
    for pat, fn in _POS_PATTERNS:
        pm = pat.search(name)
        if pm:
            axle, side, sub = fn(pm)
            return Position(axle, side, sub), comp
    return None


def discover(names: List[str]) -> "ChannelSet":
    """Build a ChannelSet from a list of raw filenames / CSV columns."""
    mapping: Dict[str, Tuple[Position, str]] = {}
    for n in names:
        parsed = parse_channel(n)
        if parsed:
            mapping[n] = parsed
    positions = sorted({p for p, _ in mapping.values()})
    components = sorted({c for _, c in mapping.values()},
                        key=lambda c: COMPONENTS.index(c) if c in COMPONENTS else 99)
    return ChannelSet(positions=positions, components=tuple(components), source_map=mapping)


# ChannelSet

@dataclass
class ChannelSet:
    positions: List[Position]
    components: Tuple[str, ...]
    source_map: Dict[str, Tuple[Position, str]]      # original name -> (pos, comp)

    @property
    def n_axles(self) -> int:
        return max((p.axle for p in self.positions), default=0)

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def has_duals(self) -> bool:
        return any(p.sub for p in self.positions)

    def axles(self) -> Dict[int, List[Position]]:
        out: Dict[int, List[Position]] = {}
        for p in sorted(self.positions):
            out.setdefault(p.axle, []).append(p)
        return out

    def force_channels(self) -> List[str]:
        return [p.channel(c) for p in sorted(self.positions)
                for c in FORCE_COMPONENTS if c in self.components]

    def canonical_rename(self) -> Dict[str, str]:
        """original name -> canonical channel name (e.g. WFT_Fx_fl -> A1L_Fx)."""
        return {orig: pos.channel(comp) for orig, (pos, comp) in self.source_map.items()}

    def summary(self) -> str:
        cfg = f"{self.n_axles}-axle"
        if self.has_duals:
            cfg += " (dual wheels)"
        return f"{self.n_positions} positions | {cfg} | components {','.join(self.components)}"