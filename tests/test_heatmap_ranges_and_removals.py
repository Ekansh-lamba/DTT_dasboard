"""Round 8 (2026-10-28): the standard heatmap bin ranges and their one source,
and the removal of the rainflow, PSD and RF Compare features.

Heatmaps: config.HEATMAP_BIN_RANGES_DAN is read by the GUI's bin-range boxes,
the pipeline's correlation heatmaps and hexbins, and the report's Fx & Fy
combination heatmap. A vehicle preset may declare its own bins; a range the
user sets for a study in the GUI wins over both, for that study's later runs.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtt.analysis.heatmaps import (
    HEATMAP_BINS_FILE, _comp_bins, _hist2d_pct, bins_from_ranges, clear_user_bin_ranges,
    fx_fy_combination_panel, generate_heatmaps, heatmap_bin_ranges, load_user_bin_ranges,
    resolve_bin_ranges, save_user_bin_ranges,
)
from dtt.config import HEATMAP_BIN_RANGES_DAN, RunConfig
from dtt.run_channels import build_run_channels
from dtt.vehicles import PRESETS, get_preset

REPO = Path(__file__).resolve().parent.parent
PV = get_preset("Passenger Vehicle (4x2)")


def _cfg(tmp_path, tyre=None) -> RunConfig:
    cfg = RunConfig(output_dir=tmp_path, study_name="hm")
    cfg.run_channels = build_run_channels(
        ["FR_Fx_2", "FR_Fy_2", "FR_Fz_2", "RR_Fx_1", "RR_Fy_1", "RR_Fz_1"])
    cfg.tyre = tyre
    return cfg


# ------------------------------------------------------------ the standard

def test_standard_ranges_are_the_gui_panel_values():
    assert HEATMAP_BIN_RANGES_DAN == {"Fx": (-300.0, 300.0, 100.0),
                                      "Fy": (-300.0, 300.0, 100.0),
                                      "Fz": (100.0, 1000.0, 100.0)}


def test_standard_edges_are_exact():
    bins = _comp_bins(None)
    np.testing.assert_array_equal(bins["Fx"], np.arange(-300.0, 301.0, 100.0))
    np.testing.assert_array_equal(bins["Fy"], np.arange(-300.0, 301.0, 100.0))
    np.testing.assert_array_equal(bins["Fz"], np.arange(100.0, 1001.0, 100.0))


def test_pv_preset_uses_the_standard_bins():
    assert PV.tyre.heatmap_bins_dan is None
    assert heatmap_bin_ranges(PV.tyre) == heatmap_bin_ranges()


def test_other_presets_keep_their_own_six_bin_ranges():
    """The per-vehicle override: a truck keeps the bins it always had (its
    tyre ranges in six equal bins), not the passenger-car ±300 daN."""
    for preset in PRESETS:
        if preset.vclass == "PV":
            continue
        t = preset.tyre
        bins = _comp_bins(t)
        for comp, (lo, hi) in (("Fx", t.fx_range_dan), ("Fy", t.fy_range_dan),
                               ("Fz", t.fz_range_dan)):
            np.testing.assert_allclose(bins[comp], np.linspace(lo, hi, 7), atol=1e-9)


def test_gui_page_defaults_come_from_the_shared_constant():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    import dtt.config as config
    from gui.models.repository import StudyRepository
    from gui.pages.heatmaps_page import HeatmapsPage

    app = QApplication.instance() or QApplication([])
    page = HeatmapsPage(StudyRepository())
    shown = {c: page._read_bin_range(c) for c in ("Fx", "Fy", "Fz")}
    assert shown == config.HEATMAP_BIN_RANGES_DAN


# ---------------------------------------------------- per-study GUI choice

def test_resolution_order_user_then_preset_then_standard(tmp_path):
    truck = get_preset("Heavy Truck (6x4 tandem)").tyre
    cfg = _cfg(tmp_path, tyre=PV.tyre)
    assert resolve_bin_ranges(cfg) == (heatmap_bin_ranges(), "standard")

    cfg.tyre = truck
    ranges, source = resolve_bin_ranges(cfg)
    assert ranges == heatmap_bin_ranges(truck) and source.startswith("vehicle preset")

    user = {"Fx": (-200.0, 200.0, 50.0), "Fy": (-400.0, 400.0, 200.0), "Fz": (0.0, 1200.0, 300.0)}
    save_user_bin_ranges(cfg.run_output_dir, user)
    ranges, source = resolve_bin_ranges(cfg)
    assert ranges == user and HEATMAP_BINS_FILE in source

    clear_user_bin_ranges(cfg.run_output_dir)
    assert load_user_bin_ranges(cfg.run_output_dir) is None


def test_unreadable_user_file_falls_back(tmp_path):
    cfg = _cfg(tmp_path)
    (cfg.run_output_dir / HEATMAP_BINS_FILE).write_text('{"Fx": [1, 0, 5]}', encoding="utf-8")
    assert resolve_bin_ranges(cfg) == (heatmap_bin_ranges(), "standard")


def _frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for w in ("FR_{}_2", "RR_{}_1"):
        cols[w.format("Fx")] = rng.normal(0, 120, n)
        cols[w.format("Fy")] = rng.normal(0, 120, n)
        cols[w.format("Fz")] = rng.normal(550, 150, n)
    return pd.DataFrame(cols)


def test_pipeline_heatmaps_follow_the_users_range(tmp_path, caplog):
    import matplotlib.pyplot as plt
    seen = []
    orig = plt.Axes.set_xticks
    cfg = _cfg(tmp_path, tyre=PV.tyre)
    user = {"Fx": (-200.0, 200.0, 50.0), "Fy": (-400.0, 400.0, 200.0), "Fz": (0.0, 1200.0, 300.0)}
    save_user_bin_ranges(cfg.run_output_dir, user)

    def spy(self, ticks, *a, **k):
        seen.append(np.asarray(ticks, dtype=float))
        return orig(self, ticks, *a, **k)

    with caplog.at_level(logging.INFO):
        plt.Axes.set_xticks = spy
        try:
            generate_heatmaps(_frame(), cfg)
        finally:
            plt.Axes.set_xticks = orig
    assert "Fx -200..200 step 50" in caplog.text
    want = bins_from_ranges(user)
    assert any(np.array_equal(t, want["Fy"]) for t in seen)      # Fx-Fy / Fz-Fy x axis
    assert any(np.array_equal(t, want["Fx"]) for t in seen)      # Fz-Fx x axis
    for name in ("heatmap_fx_fy_all", "heatmap_fz_fy_all", "heatmap_fz_fx_all",
                 "heatmap_hexbin_FR", "heatmap_hexbin_RR"):
        assert (cfg.figures_dir / f"{name}.png").exists()


def test_report_combination_heatmap_uses_the_same_bins_in_daN(tmp_path):
    from dtt.reporting.result_figures import generate_result_figures, figure_path
    cfg = _cfg(tmp_path, tyre=PV.tyre)
    generate_result_figures(_frame(), cfg)
    assert figure_path(cfg, "combo_a").exists()

    rc = cfg.run_channels
    bins = _comp_bins(PV.tyre)
    H, fy_e, fx_e = fx_fy_combination_panel(_frame(), rc, "FR", bins["Fy"], bins["Fx"])
    np.testing.assert_array_equal(fy_e, np.arange(-300.0, 301.0, 100.0))
    np.testing.assert_array_equal(fx_e, np.arange(-300.0, 301.0, 100.0))
    assert H.shape == (6, 6)


# ------------------------------------------------- out-of-range behaviour

def test_out_of_range_samples_are_dropped_not_clipped():
    """np.histogram2d leaves samples outside the edges uncounted, and the
    cell percentages are shares of the in-range samples only."""
    edges = np.array([0.0, 1.0, 2.0])
    x = np.array([0.5, 1.5, 1.5, 5.0, -3.0])
    y = np.array([0.5, 0.5, 1.5, 0.5, 0.5])
    H, _, _ = _hist2d_pct(x, y, edges, edges)
    np.testing.assert_allclose(H, [[100 / 3, 0.0], [100 / 3, 100 / 3]])
    assert H.sum() == pytest.approx(100.0)


# --------------------------------------------------- old configs still load

def test_old_settings_with_removed_keys_load_with_one_warning(tmp_path, caplog):
    old = {"study_name": "old", "output_dir": tmp_path, "miner_exponent": 8.0,
           "rainflow_bins": 16, "psd_nperseg": 1024, "remove_stops": True}
    with caplog.at_level(logging.WARNING):
        cfg = RunConfig.from_dict(old)
    assert cfg.remove_stops is True and not hasattr(cfg, "miner_exponent")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "miner_exponent" in warnings[0].getMessage()


def test_old_provenance_with_rainflow_keys_loads(tmp_path):
    import json
    from dtt.provenance import check_stale_provenance, load_provenance
    (tmp_path / "run_provenance.json").write_text(json.dumps(
        {"study_name": "old", "famos_applied": True, "miner_exponent": 8.0,
         "rainflow": {"bins": 16}, "psd": {"nperseg": 400}}), encoding="utf-8")
    assert load_provenance(tmp_path)["miner_exponent"] == 8.0
    check_stale_provenance(tmp_path)                 # warns, never raises


# ------------------------------------------------------------ the removals

def test_nothing_in_the_app_imports_legacy():
    for root in ("dtt", "gui", "famos"):
        for py in (REPO / root).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not any(n.split(".")[0] == "legacy" for n in names), py
                assert not any(n in ("rainflow", "dtt.analysis.rainflow", "dtt.analysis.psd")
                               for n in names), py


def test_cli_and_gui_carry_no_rainflow_or_psd_controls():
    pytest.importorskip("PySide6")
    from gui.main_window import NAV_ITEMS, _STUDY_PAGES
    from gui.workers.pipeline_worker import RunRequest, STAGES
    keys = [k for k, _, _ in NAV_ITEMS]
    assert "rainflow" not in keys and "psd" not in keys
    assert not ({"rainflow", "psd"} & _STUDY_PAGES)
    assert "Rainflow" not in STAGES
    assert "--miner" not in RunRequest(csv_path=Path("x.csv")).to_cmd()
    assert not hasattr(RunRequest(), "miner")
    src = (REPO / "dtt" / "pipeline.py").read_text(encoding="utf-8")
    assert "--miner" not in src and "generate_psd" not in src and "generate_rainflow" not in src
