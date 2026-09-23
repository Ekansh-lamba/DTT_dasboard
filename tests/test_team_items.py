"""Channel naming, box plots for every parameter, and the session join shown.

These are the team list's steps 2, 3 and 6.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from dtt.channel_names import (
    canonical_name, channel_unit, display_name, load_names, save_names,
)


# ------------------------------------------------------------ channel naming

@pytest.mark.parametrize("src,std", [
    ("FR_Fx_2", "FR_Fx"), ("FR_Fx", "FR_Fx"), ("fr_fz", "FR_Fz"),
    ("RR_My_1", "RR_My"), ("A3LO_Fz", "A3LO_Fz"),
    ("Latacc", "Latacc"), ("Vehicle_Speed", "Vehicle_Speed"),
])
def test_every_spelling_of_a_channel_has_one_standard_name(src, std):
    assert canonical_name(src) == std


def test_display_name_is_keyed_by_the_standard_name(tmp_path):
    """One entry covers every export of the channel."""
    f = tmp_path / "names.json"
    save_names({"FR_Fx_2": "Front Right Fx", "Latacc": "Lateral accel"}, f)
    names = load_names(f)
    assert names == {"FR_Fx": "Front Right Fx", "Latacc": "Lateral accel"}
    assert display_name("FR_Fx_2", names) == "Front Right Fx"
    assert display_name("FR_Fx", names) == "Front Right Fx"
    assert display_name("FR_Fy_2", names) == "FR_Fy"          # no name set


def test_blank_and_default_names_are_not_stored(tmp_path):
    f = tmp_path / "names.json"
    save_names({"FR_Fx": "  ", "FR_Fy_2": "FR_Fy", "FR_Fz": "Steer R Fz"}, f)
    assert load_names(f) == {"FR_Fz": "Steer R Fz"}


def test_units_follow_the_channel_not_the_magnitude():
    assert channel_unit("FR_My_2") == "daN·m"
    assert channel_unit("FR_Fz_2") == "daN"
    assert channel_unit("Latacc") == "m/s²"
    assert channel_unit("Vehicle_Speed") == "km/h"


# --------------------------------------------------------------- box plots

def _frame(n=4_000):
    rng = np.random.default_rng(0)
    cols = {"Time": np.arange(n) / 100.0}
    for w in ("FR", "RR"):
        for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            cols[f"{w}_{c}_{1 if w == 'RR' else 2}"] = rng.normal(0, 50, n)
    cols.update(Latacc=rng.normal(0, 1, n), Vehicle_Speed=rng.uniform(0, 80, n),
                Latitude=np.full(n, 18.5), Distance=np.cumsum(np.ones(n)),
                AnglePitch=np.zeros(n), FR_Angle_2=np.arange(n) % 360.0)
    return pd.DataFrame(cols)


def test_every_component_and_every_parameter_gets_a_box_plot(tmp_path):
    from dtt.analysis.boxplots import generate_boxplots, other_parameters
    from dtt.run_channels import build_run_channels

    df = _frame()
    cfg = types.SimpleNamespace(figures_dir=tmp_path,
                                run_channels=build_run_channels(list(df.columns)))
    written = generate_boxplots(df, cfg)
    for comp in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
        assert f"boxplot_{comp}.png" in written, comp       # moments too
    assert "boxplot_parameters.png" in written
    for name in written:
        assert (tmp_path / name).stat().st_size > 0

    wheel = [c for c in df.columns if c[:2] in ("FR", "RR") and "Angle" not in c]
    params = other_parameters(df, wheel)
    assert set(params) == {"Latacc", "Vehicle_Speed"}
    # not parameters: the clock, position, a running total, a channel that
    # never moves, and the wheel's rotation angle
    for gone in ("Time", "Latitude", "Distance", "AnglePitch", "FR_Angle_2"):
        assert gone not in params


# ------------------------------------------------------------ session join

def _joined_meta():
    return {"sessions": 3, "seam_times_s": [600.0, 1500.0],
            "session_durations_s": [600.0, 900.0, 400.0],
            "session_sources": ["C:/data/day1", "C:/data/day2a", "C:/data/day2b"]}


def test_session_info_reads_metadata_or_provenance_and_ignores_single_runs():
    from dtt.analysis.sessions_plot import session_info, session_spans
    assert session_info({"sessions": 1}) is None
    assert session_info(None, {"sessions": 1, "seam_times_s": []}) is None
    info = session_info(None, _joined_meta())
    assert info["sessions"] == 3
    assert session_spans(info) == [(0.0, 600.0), (600.0, 1500.0), (1500.0, 1900.0)]


def test_joined_study_gets_a_seam_figure_and_a_report_slide(tmp_path):
    from dtt.analysis.sessions_plot import (SESSIONS_FIGURE,
                                            generate_sessions_figure, session_info)
    from dtt.reporting.report_builder import _add_sessions_slide

    df = pd.DataFrame({"Time": np.arange(0, 1900, 0.5),
                       "Vehicle_Speed": np.abs(np.sin(np.arange(3800) / 50)) * 60})
    info = session_info(_joined_meta())
    path = generate_sessions_figure(df, info, tmp_path)
    assert path == tmp_path / SESSIONS_FIGURE and path.stat().st_size > 0
    assert generate_sessions_figure(df, None, tmp_path / "x") is None

    from dtt.reporting.report_builder import _Deck, _new_presentation
    deck = _Deck(_new_presentation(), footer="test")
    _add_sessions_slide(deck, info, tmp_path)
    slide = deck.prs.slides[0]
    text = "\n".join(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)
    assert "3 joined" in text and "day2a" in text and "600 – 1500 s" in text
    assert any(sh.shape_type == 13 for sh in slide.shapes)      # the figure


def test_provenance_keeps_the_seams():
    from dtt.provenance import build_provenance
    cfg = types.SimpleNamespace(study_name="s", famos_applied=True, famos_mode=True,
                                sampling_rate=100.0)
    rec = build_provenance(cfg, _joined_meta())
    assert rec["sessions"] == 3 and rec["seam_times_s"] == [600.0, 1500.0]
    assert rec["session_sources"][1] == "C:/data/day2a"
