"""
engine/__init__.py

Exports the public API of the DTT WFT engine.
Import from here rather than individual sub-modules where possible.
"""

from .io_loader import load_csv_chunked, detect_sample_rate, rebuild_time_axis, validate_data
from .channel_map import build_channel_map, find_channel_strict
from .histograms import compute_force_histograms
from .heatmaps import compute_heatmap, compute_all_heatmaps
from .trip_stats import compute_trip_stats
from .gg_severity import compute_gg, compute_force_severity
from .fatigue import compute_fatigue_summary
from .box_distance import compute_box_stats, compute_distance_distribution

__all__ = [
    "load_csv_chunked",
    "detect_sample_rate",
    "rebuild_time_axis",
    "validate_data",
    "build_channel_map",
    "find_channel_strict",
    "compute_force_histograms",
    "compute_heatmap",
    "compute_all_heatmaps",
    "compute_trip_stats",
    "compute_gg",
    "compute_force_severity",
    "compute_fatigue_summary",
    "compute_box_stats",
    "compute_distance_distribution",
]
