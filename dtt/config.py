from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent

MANDATORY_CHANNELS: List[str] = [
    "FL_Fx", "FL_Fy", "FL_Fz",
    "FR_Fx", "FR_Fy", "FR_Fz",
    "RL_Fx", "RL_Fy", "RL_Fz",
    "RR_Fx", "RR_Fy", "RR_Fz",
]

OPTIONAL_MOMENT_CHANNELS: List[str] = [
    "FL_Mx", "FL_My", "FL_Mz",
    "FR_Mx", "FR_My", "FR_Mz",
    "RL_Mx", "RL_My", "RL_Mz",
    "RR_Mx", "RR_My", "RR_Mz",
]

WHEEL_GROUPS: Dict[str, List[str]] = {
    "FL": ["FL_Fx", "FL_Fy", "FL_Fz"],
    "FR": ["FR_Fx", "FR_Fy", "FR_Fz"],
    "RL": ["RL_Fx", "RL_Fy", "RL_Fz"],
    "RR": ["RR_Fx", "RR_Fy", "RR_Fz"],
}

WHEEL_COLORS: Dict[str, Dict[str, str]] = {
    "FL": {"pri": "#2E86DE", "alt": "#74B9FF", "dark": "#0A3D7A"},
    "FR": {"pri": "#E74C3C", "alt": "#FF8A80", "dark": "#922B21"},
    "RL": {"pri": "#27AE60", "alt": "#6FCF97", "dark": "#145A32"},
    "RR": {"pri": "#8E44AD", "alt": "#C39BD3", "dark": "#4A235A"},
}

CHAN_COLORS: Dict[str, str] = {
    ch: WHEEL_COLORS[ch[:2]]["pri"]
    for ch in MANDATORY_CHANNELS
}

FORCE_RANGES_DAN: Dict[str, Tuple[float, float]] = {
    "Fx": (-300.0, 300.0),
    "Fy": (-300.0, 300.0),
    "Fz": (100.0,  1000.0),
}

BOX_YLIMS: Dict[str, Tuple[float, float]] = {
    "Fx": (-1200.0, 1200.0),
    "Fy": (-1200.0, 1200.0),
    "Fz": (5000.0,  13000.0),
}

FZ_HEATMAP_BINS_DAN = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0]
FXY_HEATMAP_BIN_STEP_DAN = 100.0
FXY_HEATMAP_RANGE_DAN     = 300.0

HEXBIN_EXTENT_N  = [-3000, 3000, -3000, 3000]
HEXBIN_GRIDSIZE  = 80

PERCENTILES: List[int] = [80, 90, 95]

PLOT_COLORS = {
    "bg":       "#0D1B2A",
    "panel":    "#1B3A5C",
    "accent":   "#E8862A",
    "success":  "#27AE60",
    "danger":   "#C0392B",
    "entry_bg": "#0A2540",
    "text_pri": "#FFFFFF",
    "text_sec": "#90E0EF",
    "btn_bg":   "#00B4D8",
}

RAINFLOW_BINS:   int   = 16
RAINFLOW_MINER:  float = 8.0
RAINFLOW_MAXPTS: int   = 50000

RF_STEPS  = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
RF_COLORS = [
    "#000080", "#0000FF", "#0080FF", "#00FFFF", "#00FF80",
    "#00FF00", "#80FF00", "#FFFF00", "#FF8000", "#FF0000", "#800000",
]

HIST_BINS: int = 60

FILTER_ORDER:  int   = 4
FILTER_CUTOFF: float = 10.0

DEFAULT_SAMPLING_RATE: float = 100.0

TIME_COLUMN:     str = "Time"
DISTANCE_COLUMN: str = "Dist"
SPEED_COLUMN:    str = "Vehicle_Speed"

SPEED_CANDIDATES: List[str] = [
    "Vehicle_Speed", "VehicleSpeed", "Speed",
    "Velocity_Speed2D", "Vehicle Speed", "VEHICLE_SPEED",
]

EXCLUDE_KEYWORDS: List[str] = [
    "lat", "lon", "latitude", "longitude", "long",
    "latacc", "longacc", "acc", "speed", "dist", "time",
]

N_TO_DAN_THRESHOLD: float = 1000.0
N_TO_DAN_FACTOR:    float = 10.0

OUTLIER_LOW_PCTILE:  float = 1.0
OUTLIER_HIGH_PCTILE: float = 99.0

GPS_SPIKE_THRESHOLD_M: float = 200.0

OUTPUTS_DIR = BASE_DIR / "dtt" / "outputs"

FIGURE_DPI: int = 150

PPTX_SLIDE_WIDTH_EMU:  int = 9144000
PPTX_SLIDE_HEIGHT_EMU: int = 5143500


@dataclass
class RunConfig:
    csv_path:       Optional[Path] = None
    raw_folder:     Optional[Path] = None      # imc STUDIO .raw folder (alt source)
    raw_files:      Optional[list] = None       # explicit subset of .raw files
    vehicle_type:   str   = ""                 # chosen preset name ("" = auto-detect)
    vehicle_name:   str   = "Vehicle"
    study_name:     str   = ""
    sampling_rate:  float = DEFAULT_SAMPLING_RATE
    filter_order:   int   = FILTER_ORDER
    filter_cutoff:  float = FILTER_CUTOFF
    apply_filter:   bool  = True
    miner_exponent: float = RAINFLOW_MINER
    output_dir:     Path  = field(default_factory=lambda: OUTPUTS_DIR)
    run_channels:   object = None      # ChannelSet-derived config, built at run time
    tyre:           object = None      # resolved TyreParams for the vehicle type

    def __post_init__(self) -> None:
        import datetime
        if not self.study_name:
            self.study_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_output_dir = Path(self.output_dir) / self.study_name
        self.figures_dir    = self.run_output_dir / "figures"
        self.logs_dir       = self.run_output_dir / "logs"
        for d in (self.run_output_dir, self.figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
