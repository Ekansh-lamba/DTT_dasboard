"""Score the FAMOS operators against output captured from a licensed imc FAMOS.

The capture in ``golden_corpus/famos_out/`` was made once, inside a 30-day imc
FAMOS trial that has since expired. It cannot be retaken, which is exactly why
it is worth a test: without one, "our smo still matches FAMOS" degrades from a
measurement into a memory of a measurement, and a future edit to the kernel
would be caught by nothing.

What is compared is ``FR_Fx_2.csv`` — a genuine FAMOS ASCII export carrying,
for each of six real WFT force channels, the raw signal beside FAMOS's own
``_smo`` (smo(0.1)) and ``_red`` (smo then red(10)) results, plus ``_rawred``
(red alone, no smoothing).

Two limits are deliberate:

* The file is gitignored (2 GB), so every test here skips when it is absent
  rather than failing. A skip is honest; a pass on missing evidence is not.
* The corpus covers *force* channels. Moments are not separately captured, and
  do not need to be: ``test_moment_and_force_are_conditioned_the_same_way`` in
  ``test_preprocess_moment_channels.py`` pins moments to byte-identical
  treatment, so validating the force path validates the operator the moment
  path runs.

The ASCII export prints ~6 significant figures, so on signals of 45,000-67,000 N
the quantisation step is ~0.05 N. That, not our arithmetic, is the error floor
these tolerances sit against — the binary ``.dat`` capture would score tighter,
but it is in the newer imc3 container that ``read_famos_all`` does not parse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from famos import ops

GOLDEN = Path(__file__).resolve().parents[1] / "golden_corpus" / "famos_out" / "FR_Fx_2.csv"
FS = 1000.0
ROWS = 200_000          # ~200 s; enough for a decisive score, quick to read
EDGE = 2_000            # drop the smo transient at both ends
CHANNELS = ("FR_Fx_2", "FR_Fy_2", "FR_Fz_2", "RR_Fx_1", "RR_Fy_1", "RR_Fz_1")

# The export's own rounding floor, not an arithmetic allowance.
MAX_ABS_ERR = 1.0       # N, against |signal| ~ 5e4
MIN_R = 0.999_999_99

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists(),
    reason=f"golden corpus not present ({GOLDEN.name} is gitignored, 2 GB)")


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    # row 1 is the units line ("s, N, N, ...")
    df = pd.read_csv(GOLDEN, skiprows=[1], nrows=ROWS)
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(df[name], errors="coerce").to_numpy(float)


def test_export_is_at_the_expected_rate(golden):
    t = _col(golden, "x")
    fs = 1.0 / np.median(np.diff(t[:1000]))
    assert fs == pytest.approx(FS, rel=1e-6)


@pytest.mark.parametrize("ch", CHANNELS)
def test_smo_matches_famos(ch, golden):
    """smo(0.1) against FAMOS's own smo output, sample for sample."""
    ours = ops.smo(_col(golden, ch), 0.1, FS)
    ref = _col(golden, f"{ch}_smo")
    a, b = ours[EDGE:-EDGE], ref[EDGE:-EDGE]
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 100_000
    err = np.abs(a[m] - b[m])
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    assert err.max() < MAX_ABS_ERR, f"{ch}: max|err| {err.max():.4g} N"
    assert r > MIN_R, f"{ch}: r = {r:.12f}"


@pytest.mark.parametrize("ch", CHANNELS)
def test_smo_then_red_matches_famos(ch, golden):
    """The full WFT force chain: smo(0.1) -> red(10)."""
    ours = ops.red(ops.smo(_col(golden, ch), 0.1, FS), 10)
    ref = _col(golden, f"{ch}_red")
    ref = ref[np.isfinite(ref)]                 # FAMOS writes it every 10th row
    n = min(ours.size, ref.size)
    e = EDGE // 10
    a, b = ours[e:n - e], ref[e:n - e]
    err = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1])
    assert err.max() < MAX_ABS_ERR, f"{ch}: max|err| {err.max():.4g} N"
    assert r > MIN_R, f"{ch}: r = {r:.12f}"


@pytest.mark.parametrize("ch", ("FR_Fz_2", "RR_Fz_1"))
def test_red_alone_is_bit_exact(ch, golden):
    """red() with no smoothing must be *exactly* FAMOS's decimation.

    This is the one comparison with no rounding excuse available: both sides
    are copies of stored samples, so any difference at all would mean the
    decimation phase (start at index 0, stride 10) disagrees with FAMOS.
    """
    ours = ops.red(_col(golden, ch), 10)
    ref = _col(golden, f"{ch}_rawred")
    ref = ref[np.isfinite(ref)]
    n = min(ours.size, ref.size)
    assert n > 10_000
    np.testing.assert_array_equal(ours[:n], ref[:n])
