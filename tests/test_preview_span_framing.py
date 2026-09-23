"""Span selection and y-framing on the Preprocess preview.

Two bugs are pinned here.

**The frame hid real samples.** The y-limits were robust percentiles of the
*drawn envelope* -- P0.2/P99.8 of the processed, P0.5/P99.5 of the raw. A
percentile over drawn points drops the few most extreme points on each side,
and what one point stands for grows with the span (one sample at 10 s, ~67 at
600 s, ~617 at the full recording), so the tallest real peaks were cut off,
by a different amount at every span. Audited over every channel of a real
study: 28 of 38 channels had samples outside the frame, the conditioned trace
up to 4.2 frame-heights beyond the edge.

**The window was chosen over one time axis.** ``_view_range`` took the raw's
Time column when there was one, so a processed trace running past the raw's
end lost its tail at "Full recording", and "the span covers the recording"
was tested against the wrong length. Latent on the reference study (raw and
processed start together and the raw is the longer), live on any other.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="the preview lives on a Qt page")

from gui.pages.preprocess_page import (                          # noqa: E402
    _frame_note, recording_extent, span_window,
)


# ---------------------------------------------------------------- the window

def test_extent_covers_raw_longer_than_processed():
    raw_t = np.arange(0, 5552.0, 0.01)
    proc_t = np.arange(0, 5547.84, 0.01)          # sanitisation dropped the tail
    assert recording_extent(raw_t, proc_t) == pytest.approx((0.0, raw_t[-1]))


def test_extent_covers_processed_longer_than_raw():
    """The case the old single-axis window got wrong."""
    raw_t = np.arange(0, 100.0, 0.01)
    proc_t = np.arange(0, 120.0, 0.01)
    start, end = recording_extent(raw_t, proc_t)
    assert end == pytest.approx(proc_t[-1])
    # full recording must reach the processed tail, not stop at the raw's end
    assert span_window(start, end, 0.0, 0.0)[1] == pytest.approx(proc_t[-1])


def test_extent_of_a_recording_that_does_not_start_at_zero():
    t = np.arange(1000.0, 1600.0, 0.01)
    start, end = recording_extent(t, None)
    assert (start, end) == pytest.approx((1000.0, t[-1]))


def test_extent_ignores_missing_axes_and_nans():
    t = np.array([np.nan, 5.0, 6.0, 7.0, np.nan])
    assert recording_extent(None, t) == (5.0, 7.0)
    assert recording_extent(None, None) == (0.0, 0.0)


def test_span_equal_to_the_recording_shows_the_recording():
    assert span_window(0.0, 600.0, 600.0, 0.5) == (0.0, 600.0)


def test_span_longer_than_the_recording_shows_the_recording():
    """A 600 s span on a 400 s recording is the recording, not 600 s of axis
    with the data crammed into one corner."""
    assert span_window(0.0, 400.0, 600.0, 1.0) == (0.0, 400.0)


def test_full_recording_ignores_the_slider():
    assert span_window(10.0, 500.0, 0.0, 0.7) == (10.0, 500.0)


@pytest.mark.parametrize("frac,expected", [
    (0.0, (100.0, 700.0)),
    (0.5, (2476.0, 3076.0)),
    (1.0, (4852.0, 5452.0)),
])
def test_window_is_exactly_the_span_and_slides_end_to_end(frac, expected):
    t0, t1 = span_window(100.0, 5452.0, 600.0, frac)
    assert t1 - t0 == pytest.approx(600.0)
    assert (t0, t1) == pytest.approx(expected)


def test_slider_is_clamped():
    assert span_window(0.0, 1000.0, 100.0, 1.7) == (900.0, 1000.0)
    assert span_window(0.0, 1000.0, 100.0, -0.3) == (0.0, 100.0)


# ------------------------------------------------------------------ the frame

def test_squash_note_is_silent_on_an_ordinary_trace():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 5_000)
    assert _frame_note(v, v.min(), v.max()) == ""


def test_squash_note_names_a_lone_extreme():
    """One far sample sets the frame; the note says so instead of the frame
    quietly dropping the sample."""
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0, 1, 5_000), [400.0]])
    note = _frame_note(v, v.min(), v.max())
    assert "far samples" in note and "%" in note


def _page_with(t, proc, raw):
    from PySide6.QtWidgets import QApplication
    from gui.models.repository import StudyRepository
    from gui.pages.preprocess_page import PreprocessPage

    app = QApplication.instance() or QApplication([])
    page = PreprocessPage(StudyRepository())
    page.resize(1300, 800)
    page.show()
    page.channel_combo.addItems(["FR_Fz_2"])
    page._pending_channel = "FR_Fz_2"
    page._apply_loaded("FR_Fz_2", t, proc, t, raw)
    return app, page


def _spans(page):
    for secs in (10.0, 30.0, 120.0, 600.0, 0.0):
        page.window_combo.setCurrentIndex(page.window_combo.findData(secs))
        page._on_window_changed()
        for sl in ((0, 500, 1000) if secs else (0,)):
            page.window_slider.setValue(sl)
            page._update()
            yield secs, sl


def test_frame_never_cuts_the_conditioned_trace_and_reports_the_raw():
    """The contract, on the real page at every span: the conditioned trace is
    always inside the frame; a raw sample outside it is always *reported* in
    the caption -- the frame reaches at most one conditioned-height past the
    conditioned trace, so an acquisition glitch 20x the body cannot squash the
    channel flat, and cannot vanish either."""
    from famos import ops

    rng = np.random.default_rng(2)
    t = np.arange(0, 3000.0, 0.01)
    raw = 800 + 60 * np.sin(t / 30) + rng.normal(0, 25, t.size)
    raw[150_000] = 4_000.0                        # a 1-sample glitch, 50x the body
    proc = ops.smo(raw, 0.1, 100.0)
    app, page = _page_with(t, proc, raw)
    saw_marker = False
    for secs, sl in _spans(page):
        x0, x1 = page.canvas.ax.get_xlim()
        y0, y1 = page.canvas.ax.get_ylim()
        m = (t >= x0) & (t <= x1)
        assert proc[m].min() >= y0 and proc[m].max() <= y1, (secs, sl)
        outside = (raw[m] < y0) | (raw[m] > y1)
        if outside.any():
            saw_marker = True
            assert "beyond the frame" in page.range_label.text(), (secs, sl)
        if secs == 0.0:
            assert (x0, x1) == pytest.approx((t[0], t[-1]))
            assert not page.window_slider.isEnabled()
        else:
            assert x1 - x0 == pytest.approx(secs)
            assert page.window_slider.isEnabled()
    assert saw_marker, "the glitch never left the frame -- the test proves nothing"
    app.processEvents()


def test_a_real_peak_within_reach_is_drawn_not_marked():
    """A genuine raw peak up to one body-height past the conditioned trace is
    inside the frame -- only far excursions become edge markers."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 600.0, 0.01)
    proc = 800 + 50 * np.sin(t / 10)             # body 750..850, height 100
    raw = proc + rng.normal(0, 5, t.size)
    raw[30_000] = 920.0                          # 70 above the body: in reach
    app, page = _page_with(t, proc, raw)
    page.window_combo.setCurrentIndex(page.window_combo.findData(0.0))
    page._on_window_changed()
    page._update()
    y0, y1 = page.canvas.ax.get_ylim()
    assert raw.max() <= y1
    assert "beyond the frame" not in page.range_label.text()
    app.processEvents()


def test_the_conditioned_trace_is_drawn_as_a_band_when_reduced():
    """At full-recording zoom the red is a band on top of the blue, as FAMOS
    draws a second trace -- not a bucket median, which over ~600 samples
    barely moves and read as over-filtered. The band is each bucket's P5..P95,
    so it is visibly shorter than the blue min..max rather than covering it."""
    from famos import ops
    from gui.pages.preprocess_page import _PROC_COLOR

    rng = np.random.default_rng(4)
    t = np.arange(0, 3000.0, 0.01)
    raw = 800 + 80 * np.sin(t / 5) + rng.normal(0, 40, t.size)
    proc = ops.smo(raw, 0.1, 100.0)
    app, page = _page_with(t, proc, raw)
    page.window_combo.setCurrentIndex(page.window_combo.findData(0.0))
    page._on_window_changed()
    page._update()
    import matplotlib.colors as mc
    lines = page.canvas.ax.lines
    red = [l for l in lines if mc.to_hex(l.get_color()) == mc.to_hex(_PROC_COLOR)]
    blue = [l for l in lines if mc.to_hex(l.get_color()) != mc.to_hex(_PROC_COLOR)
            and l.get_linestyle() != "None"]
    yr = np.concatenate([l.get_ydata() for l in red])
    from gui.pages.preprocess_page import _envelope_and_median
    x0, x1 = page.canvas.ax.get_xlim()
    _, ye, _, _ = _envelope_and_median(t, proc, yr.size // 2)
    # per-bucket stroke heights (pairs of low/high points): the drawn red is
    # shorter than the conditioned trace's own min..max, but still a band
    hr = np.abs(yr[1::2] - yr[0::2])
    he = np.abs(ye[1::2] - ye[0::2])
    ratio = np.nanmedian(hr) / np.nanmedian(he)
    assert 0.5 < ratio < 0.95, ratio
    # the frame still takes in the conditioned trace's full reach
    y0, y1 = page.canvas.ax.get_ylim()
    assert proc.min() >= y0 and proc.max() <= y1
    assert min(l.get_zorder() for l in red) > max(l.get_zorder() for l in blue)
    app.processEvents()
