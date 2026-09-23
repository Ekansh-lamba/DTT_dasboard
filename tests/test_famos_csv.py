"""The processed-data and statistics CSVs come off in the layout of the team's
FAMOS export, ``csv/RLDA WFT PV data sample.csv``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtt.famos_csv import (export_columns, famos_number, write_famos_csv,
                           write_stats_csv)


@pytest.mark.parametrize("value,text", [
    # every one of these is copied from the reference file
    (0.0, "0.0"), (-0.0, "-0.0"), (0.01, "1.0e-2"), (0.99, "0.99"),
    (999.96, "999.96"), (13921.44, "13921.4"), (169.672, "169.672"),
    (-4.81793e-2, "-4.81793e-2"), (9.06273e-2, "9.06273e-2"),
    (0.140855, "0.140855"), (-1.00104e-3, "-1.00104e-3"),
    (100000.0, "100000.0"), (131015.9, "131016"), (6121.62, "6121.62"),
    (float("nan"), ""), (None, ""),
])
def test_numbers_print_as_famos_prints_them(value, text):
    assert famos_number(value) == text


def test_columns_are_time_first_then_a_to_z_under_standard_names():
    cols = ["Time", "Vehicle_Speed", "RR_Fz_1", "Latacc", "FR_Mx_2", "Distance",
            "FR_Fx_2", "Altitude", "Latacc_LPF", "Latitude", "FR_Fx_2 - Copy"]
    assert list(export_columns(cols)) == [
        "Time", "Altitude", "Dist", "FR_Fx", "FR_Mx", "Latacc", "Latacc_LPF",
        "Latitude", "RR_Fz", "Vehicle_Speed"]
    assert export_columns(cols)["FR_Fx"] == "FR_Fx_2"       # first spelling wins


def _read_like_the_analyzer(path):
    df = pd.read_csv(path, skiprows=2, skipinitialspace=True, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df


def test_file_layout_units_and_round_trip(tmp_path):
    df = pd.DataFrame({"Time": [0.0, 0.01, 0.02],
                       "FR_Fz_2": [612.162, 613.0, np.nan],     # daN
                       "FR_Mx_2": [-22.3716, 1.0, 2.0],
                       "Latacc": [0.180139, -0.0323, 0.0]})
    out = write_famos_csv(df, tmp_path / "x.csv")
    raw = out.read_bytes().decode()
    lines = raw.split("\r\n")
    assert lines[0] == "" and lines[1] == ""
    assert lines[2] == ",".join(f"{c:>16}" for c in ("Time", "FR_Fz", "FR_Mx", "Latacc"))
    assert all(len(f) == 16 for f in lines[3].split(","))
    assert lines[4].split(",")[0].strip() == "1.0e-2"
    assert lines[5].split(",")[1] == " " * 16                   # missing sample
    back = _read_like_the_analyzer(out)
    assert list(back.columns) == ["Time", "FR_Fz", "FR_Mx", "Latacc"]
    assert back["FR_Fz"].iloc[0] == pytest.approx(6121.62)      # forces in N
    assert back["FR_Mx"].iloc[0] == pytest.approx(-22.3716)     # moments as recorded
    assert np.isnan(back["FR_Fz"].iloc[2])


def test_statistics_cover_every_channel_in_export_order(tmp_path):
    from dtt.analysis.statistics import channel_statistics
    df = pd.DataFrame({"Time": np.arange(100) / 100, "Vehicle_Speed": np.linspace(0, 50, 100),
                       "FR_Fz_2": np.linspace(600, 800, 100), "Latacc": np.zeros(100)})
    stats = channel_statistics(df)
    assert list(stats) == ["FR_Fz_2", "Latacc", "Vehicle_Speed"]     # keyed by column
    assert [e["name"] for e in stats.values()] == ["FR_Fz", "Latacc", "Vehicle_Speed"]
    assert stats["FR_Fz_2"]["unit"] == "daN" and stats["Vehicle_Speed"]["unit"] == "km/h"

    out = write_stats_csv(stats, tmp_path / "s.csv")
    back = _read_like_the_analyzer(out)
    assert list(back.columns)[:3] == ["Channel", "Unit", "Count"]
    assert [c.strip() for c in back["Channel"]] == ["FR_Fz", "Latacc", "Vehicle_Speed"]
    assert back["Unit"].str.strip().tolist() == ["daN", "m/s²", "km/h"]      # ANSI, as FAMOS
    assert back["P95"].iloc[0] == pytest.approx(np.percentile(np.linspace(600, 800, 100), 95),
                                                rel=1e-5)
