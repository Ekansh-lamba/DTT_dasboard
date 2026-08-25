"""Chunked execution that is bit-identical to processing the whole record.

A 11676 s record at 1000 Hz is 11.7 M samples per channel; a 40-channel study is
~3.7 GB in float64. Loading all of it is not an option, so the chain has to run
in pieces -- and a chain that gives a *different* answer in pieces is a data
corruption bug that no amount of testing downstream will catch, because the
output still looks like a plausible signal.

Chunk-boundary strategy, per operator
-------------------------------------
``FiltLP``  Recursive, so state carries naturally. `sosfilt` returns its final
            delay-line state; feeding it in as `zi` for the next chunk makes the
            stream mathematically identical to one long pass. The step-response
            initialisation applies to the FIRST chunk only, from that chunk's
            first sample -- which is the record's first sample, so it is the
            same x[0] the unchunked path uses.

``Smo``     FIR, symmetric, half-width `hw = (taps-1)//2`. Each chunk is
            processed with `hw` real samples of context on both sides, then the
            context is discarded. The subtlety that makes or breaks this: the
            manual's edge-replication rule applies to the TRUE ends of the
            record, never to a chunk boundary. Replicating at an interior
            boundary would fabricate a plateau in the middle of the signal. So
            interior boundaries take real neighbouring samples and only the
            first/last chunk see `mode="nearest"`.

``red``     A stride, so the only requirement is global index alignment: a chunk
            starting at global index `g` emits samples where `(g + i) % n == 0`.
            Tracking that offset is what stops each chunk restarting the pattern
            and shifting the output by up to n-1 samples.

Gap constraint
--------------
`FiltLP` bridges NaN gaps by linear interpolation, which needs the good sample
on BOTH sides. A gap straddling a chunk boundary would be interpolated from
different endpoints than in the unchunked pass. `StreamProcessor` therefore
requires `chunk_size > longest_gap + hw` and RAISES if a gap reaches a boundary,
rather than emitting a subtly different answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from famos import ops

__all__ = ["StreamingFiltLP", "StreamingSmo", "StreamingRed", "chunk_indices"]


def chunk_indices(n: int, chunk: int) -> Iterator[tuple[int, int, bool, bool]]:
    """Yield ``(start, stop, is_first, is_last)`` covering ``[0, n)``."""
    if chunk < 1:
        raise ValueError(f"chunk size must be >= 1, got {chunk}")
    start = 0
    while start < n:
        stop = min(start + chunk, n)
        yield start, stop, start == 0, stop == n
        start = stop


class StreamingFiltLP:
    """FiltLP across chunks, carrying the biquad delay line between calls."""

    def __init__(self, cutoff_hz: float, order: int, fs_hz: float,
                 init: str = ops.INIT_FAMOS,
                 character: int = ops.CHARACTER_BUTTERWORTH,
                 parameter: float = 0.0):
        self.sos = ops.filtlp_sos(order, cutoff_hz, fs_hz, character, parameter)
        self.init = init
        if init not in (ops.INIT_FAMOS, ops.INIT_LEGACY_ZERO_STATE):
            raise ValueError(f"FiltLP: unknown init {init!r}")
        self._zi: Optional[np.ndarray] = None

    def process(self, chunk, is_first: bool = False) -> np.ndarray:
        arr = ops.as_f64(chunk, "filtlp chunk")
        mask = np.isfinite(arr)
        if not mask.all():
            if not mask.any():
                raise ValueError("filtlp chunk is entirely NaN; cannot bridge")
            if not mask[0] or not mask[-1]:
                raise ValueError(
                    "a NaN gap reaches a chunk boundary. Interpolating it here "
                    "would use different endpoints than an unchunked pass and "
                    "silently change the result. Increase chunk_size past the "
                    "longest gap.")
            idx = np.arange(arr.size)
            work = arr.copy()
            work[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
        else:
            work = arr

        if is_first or self._zi is None:
            if self.init == ops.INIT_FAMOS:
                self._zi = sosfilt_zi(self.sos) * work[0]
            else:
                self._zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
        y, self._zi = sosfilt(self.sos, work, zi=self._zi)
        y[~mask] = np.nan
        return y


@dataclass
class StreamingSmo:
    """Smo across chunks by overlap, with edge replication only at true ends."""
    width_s: float
    fs_hz: float
    _tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    _pending: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.kernel = ops.smo_kernel(self.width_s, self.fs_hz)
        self.hw = (self.kernel.size - 1) // 2

    def run(self, chunks: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
        """Stream chunks through Smo, yielding smoothed output in order.

        A rolling buffer, not a one-chunk lookahead. The earlier design held
        back a single chunk and took its first `hw` samples as right-context,
        which silently degraded to edge replication whenever a chunk was
        shorter than `hw` -- fabricating a plateau in the middle of the record.
        A short final chunk is the normal case (record length rarely divides by
        chunk size), so that path was hit routinely.

        Output for global position `i` is emitted only once real samples up to
        `i + hw` have arrived, so replication is used strictly at the true ends.
        Memory is bounded: the buffer holds at most one input chunk plus `2*hw`.
        """
        hw = self.hw
        buf = np.empty(0, dtype=np.float64)
        base = 0            # global index of buf[0]
        emitted = 0         # number of output samples already yielded

        for chunk in chunks:
            buf = np.concatenate([buf, ops.as_f64(chunk, "smo chunk")])
            ready = base + buf.size - hw          # emit up to here safely
            if ready > emitted:
                yield self._span(buf, base, emitted, ready,
                                 is_first=(emitted == 0), is_last=False)
                emitted = ready
                keep = max(0, emitted - hw - base)   # retain hw of left context
                buf, base = buf[keep:], base + keep

        if emitted < base + buf.size:
            yield self._span(buf, base, emitted, base + buf.size,
                             is_first=(emitted == 0), is_last=True)

    def _span(self, buf: np.ndarray, base: int, lo: int, hi: int,
              is_first: bool, is_last: bool) -> np.ndarray:
        """Smooth global range [lo, hi) out of `buf`, which starts at `base`."""
        hw, h = self.hw, self.kernel
        l, r = lo - base, hi - base
        left = buf[max(0, l - hw):l]
        right = buf[r:r + hw]
        if left.size < hw:
            if not is_first:
                raise AssertionError("smo: lost left context mid-record")
            left = np.concatenate([np.full(hw - left.size, buf[0]), left])
        if right.size < hw:
            if not is_last:
                raise AssertionError("smo: lost right context mid-record")
            right = np.concatenate([right, np.full(hw - right.size, buf[-1])])

        core = buf[l:r]
        window = np.concatenate([left, core, right])
        mask = np.isfinite(window)
        filled = np.where(mask, window, 0.0)
        num = np.convolve(filled, h, mode="same")
        den = np.convolve(mask.astype(np.float64), h, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        out[~np.isfinite(out)] = np.nan
        out[~mask] = np.nan
        return out[hw:hw + core.size]


class StreamingRed:
    """red(x, n) across chunks, tracking the global index so phase is kept."""

    def __init__(self, factor: int):
        if factor < 1:
            raise ValueError(f"red: factor must be >= 1, got {factor}")
        self.factor = factor
        self._global = 0

    def process(self, chunk) -> np.ndarray:
        arr = ops.as_f64(chunk, "red chunk")
        if self.factor == 1:
            self._global += arr.size
            return arr.copy()
        # First surviving position within this chunk, given where we are in the
        # global stream. Without this the pattern restarts every chunk and the
        # output shifts by up to factor-1 samples.
        offset = (-self._global) % self.factor
        out = arr[offset::self.factor]
        self._global += arr.size
        return np.ascontiguousarray(out)
