"""
Commercial-vehicle type + tyre-parameter presets.

The company does not supply a fixed vehicle taxonomy, so this ships **standard
commercial-vehicle classes** with typical axle layouts and load ranges. Each
type expands to a set of canonical :class:`Position` objects and carries the
tyre parameters that drive histogram/heatmap axis limits, sanitisation
thresholds, and filtering defaults.

A recording's discovered :class:`ChannelSet` is matched to the closest preset by
axle/position count, or the user picks one in the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from dtt.channels import Position, ChannelSet


@dataclass
class TyreParams:
    tyre_type: str
    fz_range_dan: Tuple[float, float] = (0.0, 1000.0)
    fx_range_dan: Tuple[float, float] = (-300.0, 300.0)
    fy_range_dan: Tuple[float, float] = (-300.0, 300.0)
    lower_threshold_dan: float = 0.0       # aim 5: drop |force| below this
    outlier_pctile: Tuple[float, float] = (1.0, 99.0)   # aim 6
    filter_cutoff_hz: float = 10.0
    filter_order: int = 4

    def range_for(self, component: str) -> Tuple[float, float]:
        return {"Fx": self.fx_range_dan, "Fy": self.fy_range_dan,
                "Fz": self.fz_range_dan}.get(component, (-1000.0, 1000.0))


@dataclass
class Axle:
    id: int
    role: str = "load"          # steer | drive | load | trailer
    dual: bool = False          # dual wheels (inner/outer)

    def positions(self) -> List[Position]:
        if self.dual:
            return [Position(self.id, s, io) for s in ("L", "R") for io in ("O", "I")]
        return [Position(self.id, s) for s in ("L", "R")]


@dataclass
class VehicleType:
    name: str
    vclass: str                 # PV | LCV | MCV | HCV | BUS | TRAILER
    axles: List[Axle]
    tyre: TyreParams

    def positions(self) -> List[Position]:
        return [p for ax in self.axles for p in ax.positions()]

    @property
    def n_positions(self) -> int:
        return len(self.positions())

    @property
    def config_code(self) -> str:
        """e.g. '6x4' style descriptor from axle count."""
        return f"{2 * len(self.axles)}wheels-{len(self.axles)}axle"

    def expected_force_channels(self) -> List[str]:
        return [p.channel(c) for p in self.positions() for c in ("Fx", "Fy", "Fz")]


# Standard preset library
def _pv() -> VehicleType:
    return VehicleType("Passenger Vehicle (4x2)", "PV",
                       [Axle(1, "steer"), Axle(2, "drive")],
                       TyreParams("PV radial", (0, 1000), (-300, 300), (-300, 300), 5))

def _lcv() -> VehicleType:
    return VehicleType("Light Commercial (4x2)", "LCV",
                       [Axle(1, "steer"), Axle(2, "drive")],
                       TyreParams("LCV", (0, 1800), (-600, 600), (-600, 600), 10))

def _mcv_6x2() -> VehicleType:
    return VehicleType("Medium Truck (6x2)", "MCV",
                       [Axle(1, "steer"), Axle(2, "drive"), Axle(3, "load")],
                       TyreParams("MCV", (0, 3000), (-1000, 1000), (-1000, 1000), 20))

def _hcv_6x4() -> VehicleType:
    return VehicleType("Heavy Truck (6x4 tandem)", "HCV",
                       [Axle(1, "steer"), Axle(2, "drive", dual=True), Axle(3, "drive", dual=True)],
                       TyreParams("315/80 R22.5", (0, 4000), (-1500, 1500), (-1500, 1500), 50))

def _hcv_8x4() -> VehicleType:
    return VehicleType("Heavy Truck (8x4)", "HCV",
                       [Axle(1, "steer"), Axle(2, "steer"),
                        Axle(3, "drive", dual=True), Axle(4, "drive", dual=True)],
                       TyreParams("315/80 R22.5", (0, 4500), (-1600, 1600), (-1600, 1600), 50))

def _bus() -> VehicleType:
    return VehicleType("Bus (6x2)", "BUS",
                       [Axle(1, "steer"), Axle(2, "drive", dual=True), Axle(3, "load")],
                       TyreParams("bus radial", (0, 3500), (-1200, 1200), (-1200, 1200), 30))

def _trailer_3() -> VehicleType:
    return VehicleType("Trailer (3-axle)", "TRAILER",
                       [Axle(1, "trailer"), Axle(2, "trailer"), Axle(3, "trailer")],
                       TyreParams("trailer", (0, 4000), (-1500, 1500), (-1500, 1500), 50))


PRESETS: List[VehicleType] = [
    _pv(), _lcv(), _mcv_6x2(), _hcv_6x4(), _hcv_8x4(), _bus(), _trailer_3(),
]


def match_preset(cs: ChannelSet) -> VehicleType:
    """Best-fit preset for a discovered ChannelSet (by position/axle count)."""
    if cs.n_positions == 0:
        return PRESETS[0]
    best = min(PRESETS, key=lambda vt: (abs(vt.n_positions - cs.n_positions),
                                        abs(len(vt.axles) - cs.n_axles)))
    return best


def preset_names() -> List[str]:
    return [vt.name for vt in PRESETS]


def get_preset(name: str) -> Optional[VehicleType]:
    for vt in PRESETS:
        if vt.name == name:
            return vt
    return None