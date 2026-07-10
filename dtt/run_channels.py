"""
Runtime channel configuration — the axle-dynamic replacement for the hardcoded
``WHEEL_GROUPS`` / ``WHEEL_COLORS`` / ``CHAN_COLORS`` / ``MANDATORY_CHANNELS``.

Built from the actual columns present in a dataset via
:func:`dtt.channels.discover`, so a 2-axle car, a 6x4 truck, or a partial 2-WFT
recording all produce a valid grouping. Crucially, for the standard 4-wheel
layout this reproduces the legacy constants **exactly** (same labels, same
order, same colours), so existing output is unchanged.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dtt.config import WHEEL_COLORS
from dtt.channels import discover, ChannelSet, FORCE_COMPONENTS


def _gen_colors(n: int) -> List[Dict[str, str]]:
    """Distinct pri/alt/dark colour triplets for arbitrary position counts."""
    out: List[Dict[str, str]] = []
    for i in range(max(n, 1)):
        h = (i / max(n, 1)) % 1.0
        def hexc(s, v):
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        out.append({"pri": hexc(0.62, 0.85), "alt": hexc(0.40, 0.95), "dark": hexc(0.70, 0.45)})
    return out


@dataclass
class RunChannels:
    channel_set: ChannelSet
    wheel_groups: Dict[str, List[str]]              # label -> [Fx,Fy,Fz channels present]
    wheel_colors: Dict[str, Dict[str, str]]         # label -> {pri,alt,dark}
    chan_colors: Dict[str, str]                     # channel -> pri colour
    mandatory_channels: List[str]                   # all force channels, position order
    _label_comp: Dict[str, Dict[str, str]] = field(default_factory=dict)  # label -> {comp: channel}

    def channel_for(self, label: str, component: str) -> Optional[str]:
        return self._label_comp.get(label, {}).get(component)

    @property
    def labels(self) -> List[str]:
        return list(self.wheel_groups.keys())

    def summary(self) -> str:
        return self.channel_set.summary()


def build_run_channels(columns: List[str]) -> RunChannels:
    cs = discover([str(c) for c in columns])

    # Group original column names by position, keeping only force components.
    by_pos: Dict[object, Dict[str, str]] = {}
    for orig, (pos, comp) in cs.source_map.items():
        if comp in FORCE_COMPONENTS:
            by_pos.setdefault(pos, {})[comp] = orig

    positions = [p for p in sorted(cs.positions) if p in by_pos]
    palette = _gen_colors(len(positions))

    # Use legacy FL/FR/RL/RR labels only for a genuine 2-axle, single-wheel car,
    # so that output is byte-identical to the original pipeline. Any other layout
    # (3+ axles, dual wheels) uses clean canonical ids (A1L, A2RO, …).
    is_car = (positions and max(p.axle for p in positions) <= 2
              and not any(p.sub for p in positions)
              and all(p.legacy_label for p in positions))

    wheel_groups: Dict[str, List[str]] = {}
    wheel_colors: Dict[str, Dict[str, str]] = {}
    chan_colors: Dict[str, str] = {}
    label_comp: Dict[str, Dict[str, str]] = {}
    mandatory: List[str] = []

    for i, pos in enumerate(positions):
        label = pos.legacy_label if is_car else pos.id
        comps = by_pos[pos]
        ordered = [comps[c] for c in FORCE_COMPONENTS if c in comps]
        wheel_groups[label] = ordered
        label_comp[label] = dict(comps)
        # Reuse the legacy palette for FL/FR/RL/RR so car output is identical.
        color = WHEEL_COLORS.get(label) or palette[i]
        wheel_colors[label] = color
        for ch in ordered:
            chan_colors[ch] = color["pri"]
        mandatory.extend(ordered)

    return RunChannels(
        channel_set=cs,
        wheel_groups=wheel_groups,
        wheel_colors=wheel_colors,
        chan_colors=chan_colors,
        mandatory_channels=mandatory,
        _label_comp=label_comp,
    )