from __future__ import annotations
import logging
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WFT_RE = re.compile(r"WFT_(F[xyz]|M[xyz])_(fl|fr|rl|rr)$", re.I)

_AUX_MAP = {
    "AccelX": "Longacc", "AccelY": "Latacc",
    "Speed2D": "Speed2D", "Speed_kmph": "Vehicle_Speed",
    "PosAlt": "Altitude", "PosLat": "Latitude", "PosLon": "Longitude",
    "AngRateZ": "Yawrate",
}


@dataclass
class ImcChannel:
    name: str                 # pipeline name, e.g. "FL_Fx"
    raw_name: str             # original, e.g. "WFT_Fx_fl"
    unit: str
    factor: float
    data: np.ndarray          # calibrated physical values
    fs: float                 # sample rate (Hz)
    start_epoch: float = 0.0
    comment: str = ""


# Low-level block parsing

def _find_block(raw: bytes, key: bytes) -> int:
    return raw.find(key)


def _cm_factor_unit(raw: bytes) -> tuple[float, str]:
    """CM block: <u32 number format><double factor> ... <len-prefixed unit string>."""
    i = _find_block(raw, b"|CM1")
    if i < 0:
        return 1.0, ""
    factor = struct.unpack("<d", raw[i + 8:i + 16])[0]
    # unit is the short ascii run just before the next |CD/|CN block
    tail = raw[i:i + 64]
    m = re.search(rb"\x01\x00([A-Za-z%/\xb2\xb3]{1,6})\|C", tail)
    unit = m.group(1).decode("latin1") if m else ""
    if not np.isfinite(factor) or factor == 0:
        factor = 1.0
    return factor, unit


def _channel_name(raw: bytes) -> tuple[str, str]:
    """CN block: <flag><u16 len><name><u16 len><comment>."""
    i = _find_block(raw, b"|CN1")
    if i < 0:
        return "", ""
    p = i + 5
    nl = struct.unpack("<H", raw[p:p + 2])[0]
    name = raw[p + 2:p + 2 + nl].decode("latin1", "replace")
    q = p + 2 + nl
    cl = struct.unpack("<H", raw[q:q + 2])[0] if q + 2 <= len(raw) else 0
    comment = raw[q + 2:q + 2 + cl].decode("latin1", "replace") if 0 < cl < 64 else ""
    return name, comment


def _imc3_dtype(raw: bytes) -> str:
    """Sample type: the u32 that opens ``|CM1``, in the same numbering as a
    classic FAMOS ``CP`` key (4 = int16, 6 = int32, 7 = float32, ...)."""
    i = _find_block(raw, b"|CM1")
    if i < 0:
        raise ValueError("imc3: no |CM1 block, sample type unknown")
    code = struct.unpack("<I", raw[i + 4:i + 8])[0]
    if code not in _FAMOS_DT:
        raise ValueError(f"imc3: unknown sample type code {code} in |CM1")
    return _FAMOS_DT[code]


def _imc3_x_delta(raw: bytes) -> Optional[float]:
    """Sample interval in seconds: the double after the u32 that opens ``|CC1``."""
    i = _find_block(raw, b"|CC1")
    if i < 0:
        return None
    dx = struct.unpack("<d", raw[i + 8:i + 16])[0]
    return dx if np.isfinite(dx) and dx > 0 else None


def _imc3_payload(raw: bytes) -> bytes:
    """The channel's stored sample bytes, in order, from the explicit record framing.

    After the metadata blocks the data section is a sequence of records, each
    with a 4-byte key and a fixed or self-declared size:

    * ``|RR1`` -- 4 bytes, no body;
    * ``|RT1`` -- 32 bytes (key + 28-byte body);
    * ``|RC5`` -- key + three int32 (1, 1, payload length) + the payload: the
      samples themselves;
    * ``|Ri1`` -- key + two int32, the second a count N, followed by an index
      of N 16-byte entries: copies of the segment's ``|RT1`` record and
      ``|RC5`` headers *without* payload;
    * ``|RE1`` -- end of the data section.

    Every byte is accounted for, so an unknown key is an error, not something
    to skip: it means the layout is not the one this was written against, and
    returning a plausible-looking signal anyway is exactly how the old
    smoothness heuristic hid a 58-sample offset (PROJECT_HISTORY §4.6).
    """
    cp = _find_block(raw, b"|CP1")
    starts = [s for s in (raw.find(k, max(cp, 0)) for k in (b"|RR1", b"|RT1", b"|RC5"))
              if s >= 0]
    if not starts:
        raise ValueError("imc3: no data records found")
    i = min(starts)
    parts = []
    n = len(raw)
    while i < n:
        key = raw[i:i + 4]
        if key == b"|RC5":
            length = struct.unpack("<i", raw[i + 12:i + 16])[0]
            parts.append(raw[i + 16:i + 16 + length])
            i += 16 + length
        elif key == b"|RR1":
            i += 4
        elif key == b"|RT1":
            i += 32
        elif key == b"|Ri1":
            count = struct.unpack("<i", raw[i + 8:i + 12])[0]
            i += 12
            for _ in range(count):
                entry = raw[i:i + 4]
                if entry == b"|RT1":
                    i += 32
                elif entry == b"|RC5":
                    i += 16
                else:
                    raise ValueError(f"imc3: unexpected index entry {entry!r} at byte {i}")
        elif key == b"|RE1":
            return b"".join(parts)
        else:
            raise ValueError(f"imc3: unknown data record {key!r} at byte {i}")
    raise ValueError("imc3: data section has no |RE1 end record")


def read_raw(path: Path, fs: Optional[float] = None,
             itemsize: Optional[int] = None,
             n_force: Optional[int] = None) -> Optional[ImcChannel]:
    """Read one imc3 channel as its stored samples x calibration factor.

    Sample type, factor and sample interval all come from the file's own
    header (``|CM1``, ``|CC1``); ``fs`` / ``itemsize`` only override a header
    that lacks them. No sample is added, dropped or interpolated: sample n is
    at ``n * x_delta``.
    """
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|imc3"):
        return None
    raw_name, comment = _channel_name(raw)
    factor, unit = _cm_factor_unit(raw)
    try:
        dt = _imc3_dtype(raw)
    except ValueError:
        if itemsize is None:
            raise
        dt = "<i2" if itemsize == 2 else "<i4"
    payload = _imc3_payload(raw)
    size = np.dtype(dt).itemsize
    if len(payload) % size:
        raise ValueError(f"imc3: {len(payload)} payload bytes is not a whole "
                         f"number of {size}-byte samples")
    samples = np.frombuffer(payload, dtype=dt).astype(np.float64)
    dx = _imc3_x_delta(raw)
    rate = (1.0 / dx) if dx else fs
    if rate is None:
        raise ValueError("imc3: no sample interval in |CC1 and none given")

    return ImcChannel(
        name=_map_name(raw_name), raw_name=raw_name, unit=unit,
        factor=factor, data=samples * factor, fs=rate, comment=comment,
    )


def _map_name(raw_name: str) -> str:
    m = _WFT_RE.match(raw_name)
    if m:
        comp, pos = m.group(1), m.group(2).upper()
        comp = comp[0].upper() + comp[1].lower()      # Fx / Mz normalised
        return f"{pos}_{comp}"
    return _AUX_MAP.get(raw_name, raw_name)


# Windows leaves these behind when a channel file is duplicated; the reader
# already collapses them because it keys on the name *inside* the file, so a
# listing built from filenames has to collapse them the same way or it reports
# channels that will never appear in the output.
# A bare "(n)" needs the space Windows always puts there, so a channel legitimately
# named e.g. "Ch(1)" is left intact.
_COPY_SUFFIX_RE = re.compile(r"(?:\s*-\s*Copy(?:\s*\(\d+\))?|\s+\(\d+\))$", re.I)


def pipeline_channel_name(raw_name: str) -> str:
    """The column name a raw file's channel will carry once ingested.

    ``AccelY`` becomes ``Latacc``, ``WFT_Fx_fl`` becomes ``FL_Fx``, and a
    ``… - Copy`` duplicate resolves to the same name as its original. Lets the
    GUI list a source's channels under the names every later screen will use,
    without opening the (multi-megabyte) files.
    """
    return _map_name(_COPY_SUFFIX_RE.sub("", raw_name.strip()))


def resolve_raw_folder(folder: Path) -> Path:
    """Return the folder that actually holds the ``.raw`` channels.

    imc exports nest the channels one level down (``<export>/<timestamp>/*.raw``),
    so if the given folder has no ``.raw`` files, descend to the sub-folder that
    contains the most.
    """
    folder = Path(folder)
    if any(folder.glob("*.raw")):
        return folder
    counts: Dict[Path, int] = {}
    for p in folder.rglob("*.raw"):
        counts[p.parent] = counts.get(p.parent, 0) + 1
    return max(counts, key=counts.get) if counts else folder


# imcdbc (XML) — per-channel timing & rate

def read_imcdbc(folder: Path) -> Dict[str, tuple[float, float]]:
    """Return {raw_channel_name: (start_epoch, end_epoch)} from Storage.imcdbc."""
    dbc = Path(folder) / "Storage.imcdbc"
    spans: Dict[str, tuple[float, float]] = {}
    if not dbc.exists():
        return spans
    try:
        text = dbc.read_text(encoding="utf-8", errors="replace")
        # strip default namespace for simple tag access
        text = re.sub(r'\sxmlns="[^"]+"', "", text, count=1)
        root = ET.fromstring(text)
        for nx in root.findall("Nx"):
            name = nx.findtext("N", "")
            s = nx.findtext("S")
            e = nx.findtext("E")
            if name and s and e and not name.endswith(".raw"):
                try:
                    spans[name] = (float(s), float(e))
                except ValueError:
                    pass
    except ET.ParseError:
        pass
    return spans


# Classic imc FAMOS format (|CF,2,...) — older exports, ASCII self-describing keys

_FAMOS_DT = {1: "<u1", 2: "<i1", 3: "<u2", 4: "<i2",
             5: "<u4", 6: "<i4", 7: "<f4", 8: "<f8"}


def _parse_famos_keys(raw: bytes):
    """Yield (key, content_bytes) for each ``|KK,ver,len,<content>;`` block.

    Lengths are explicit, so the binary sample block is skipped correctly even
    though it may contain ``|``/``;`` bytes.
    """
    keys = []
    i, n = 0, len(raw)
    while i < n and raw[i:i + 1] == b"|":
        c1 = raw.find(b",", i)
        c2 = raw.find(b",", c1 + 1)
        c3 = raw.find(b",", c2 + 1)
        if -1 in (c1, c2, c3):
            break
        try:
            length = int(raw[c2 + 1:c3])
        except ValueError:
            break
        key = raw[i + 1:c1].decode("latin1")
        keys.append((key, raw[c3 + 1:c3 + 1 + length]))
        nxt = raw.find(b"|", c3 + 1 + length)
        if nxt < 0:
            break
        i = nxt
    return keys


def _cr_scaling(fields: List[bytes]) -> tuple[float, float, str]:
    """Decode a FAMOS ``CR`` block into (factor, offset, unit).

    Layout: ``<transform>,<factor>,<offset>,...,<unit>``.

    The FIRST field is the transformation flag, and it decides whether the other
    two mean anything: ``1`` says the stored counts must be scaled to reach
    physical units, ``0`` says the samples are ALREADY physical and no
    transformation applies. A flag-0 block carries ``factor=0``, so reading
    fields 1 and 2 unconditionally -- as this reader did -- multiplies the whole
    channel by zero and returns a silently empty signal.

    Both FAMOS entry points parse this block, so it lives here once: the two
    copies are what let the bug sit in one of them unnoticed.

    A malformed flag is treated as ``1`` (scale), which is the behaviour every
    channel in this project's corpus relies on; only an explicit ``0`` disables
    scaling, so a parse failure cannot blank a channel.
    """
    factor, offset = 1.0, 0.0
    try:
        transform = int(fields[0])
    except (ValueError, IndexError):
        transform = 1
    if transform != 0:
        try:
            factor, offset = float(fields[1]), float(fields[2])
        except (ValueError, IndexError):
            factor, offset = 1.0, 0.0
    unit = fields[-1].decode("latin1").strip()
    return factor, offset, unit


def _cb_extent(fields: List[bytes]) -> Optional[tuple[int, int]]:
    """``(start, nbytes)`` of the sample bytes inside the ``CS`` block, from a
    ``Cb`` key, or None when it does not describe exactly one buffer.

    ``Cb`` fields: buffer count, user-info bytes, then per buffer: buffer
    reference, ``CS`` key index, offset of the buffer in that ``CS`` block,
    buffer size, offset of the first sample in the buffer, bytes filled, ...

    The ``CS`` block can be longer than its buffer. imc's recorder pads it
    (8,296 zero bytes per channel on the 2006-08-25 reference recording), and
    reading the whole block turned that padding into 4,148 extra samples of 0 N
    at the end of every channel -- the start of the Round 9 timebase stretch.
    """
    try:
        if int(fields[0]) != 1:
            return None
        offset_in_cs = int(fields[4])
        first = int(fields[6])
        filled = int(fields[7])
    except (ValueError, IndexError):
        return None
    if filled <= 0 or offset_in_cs < 0 or first < 0:
        return None
    return offset_in_cs + first, filled


def _cs_samples(blob: bytes, cb: Optional[tuple[int, int]]) -> bytes:
    """The sample bytes of a ``CS`` block: the ``Cb`` buffer if one was given
    and fits, else the whole block (files written without ``Cb``)."""
    if cb is not None:
        start, nbytes = cb
        if start + nbytes <= len(blob):
            return blob[start:start + nbytes]
    return blob


def read_famos_all(path: Path) -> List[ImcChannel]:
    """Every channel in a FAMOS file, in file order.

    A FAMOS ``.dat`` holds as many channels as you save into it, which is what
    FAMOS itself writes when several variables are selected at once. The per-file
    reader below assumes one, because the imc raw exports this project started
    from happen to ship one channel each -- an assumption that quietly truncates
    a real FAMOS save to its first channel.

    The keys arrive as one flat stream, so channel boundaries are inferred: the
    describing keys (``CD`` sample step, ``CP`` number format, ``CR`` scaling,
    ``CN`` name) accumulate, and each ``CS`` block closes a channel using
    whatever is current. ``CD`` is not repeated per channel in every writer, so
    it persists across channels rather than resetting.
    """
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|CF"):
        return []

    out: List[ImcChannel] = []
    dx = None
    factor, offset, unit = 1.0, 0.0, ""
    dtype = "<i2"
    name = ""
    index = 0
    cb = None

    for key, content in _parse_famos_keys(raw):
        fields = content.split(b",")
        if key == "CD":
            try:
                dx = float(fields[0])
            except (ValueError, IndexError):
                pass
        elif key == "CP" and len(fields) >= 3:
            try:
                dtype = _FAMOS_DT.get(int(fields[2]), "<i2")
            except ValueError:
                pass
        elif key == "CR" and len(fields) >= 3:
            factor, offset, unit = _cr_scaling(fields)
        elif key == "CN" and len(fields) >= 5:
            name = fields[4].decode("latin1").strip()
        elif key == "Cb":
            cb = _cb_extent(fields)
        elif key == "CS":
            ci = content.find(b",")
            blob = _cs_samples(content[ci + 1:] if ci >= 0 else content, cb)
            cb = None
            if not blob or not dx:
                continue
            size = np.dtype(dtype).itemsize
            count = len(blob) // size
            if count == 0:
                continue
            samples = np.frombuffer(blob[:count * size],
                                    dtype=dtype).astype(np.float64)
            index += 1
            label = name or f"{Path(path).stem}_{index}"
            out.append(ImcChannel(
                name=_map_name(label), raw_name=label, unit=unit,
                factor=factor, data=samples * factor + offset, fs=1.0 / dx))
            # Name belongs to the channel just closed; the next must supply
            # its own or be numbered, rather than inheriting this one.
            name = ""
    return out


def read_famos(path: Path) -> Optional[ImcChannel]:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|CF"):
        return None
    dx = None
    factor, offset, unit = 1.0, 0.0, ""
    dtype = "<i2"
    name = Path(path).stem
    data_bytes = b""
    cb = None
    for key, content in _parse_famos_keys(raw):
        fields = content.split(b",")
        if key == "CD":
            try:
                dx = float(fields[0])
            except (ValueError, IndexError):
                pass
        elif key == "CP" and len(fields) >= 3:
            try:
                dtype = _FAMOS_DT.get(int(fields[2]), "<i2")
            except ValueError:
                pass
        elif key == "CR" and len(fields) >= 3:
            factor, offset, unit = _cr_scaling(fields)
        elif key == "CN" and len(fields) >= 5:
            name = fields[4].decode("latin1").strip()
        elif key == "Cb":
            cb = _cb_extent(fields)
        elif key == "CS":
            ci = content.find(b",")
            data_bytes = _cs_samples(content[ci + 1:] if ci >= 0 else content, cb)

    if not data_bytes or not dx:
        return None
    size = np.dtype(dtype).itemsize
    count = len(data_bytes) // size
    samples = np.frombuffer(data_bytes[:count * size], dtype=dtype).astype(np.float64)
    return ImcChannel(name=_map_name(name), raw_name=name, unit=unit,
                      factor=factor, data=samples * factor + offset, fs=1.0 / dx)
def _detect_format(files) -> str:
    for f in files:
        head = f.read_bytes()[:8]
        if head.startswith(b"|imc3"):
            return "imc3"
        if head.startswith(b"|CF"):
            return "famos"
    return "unknown"


# Files/folder → aligned DataFrame

def _read_channels(files, spans) -> List[ImcChannel]:
    files = [Path(f) for f in files]
    channels: List[ImcChannel] = []

    if _detect_format(files) == "famos":
        for f in files:
            try:
                ch = read_famos(f)
            except Exception:
                ch = None
            if ch is not None and ch.data.size:
                channels.append(ch)
        return channels
    for f in files:
        if not f.read_bytes()[:8].startswith(b"|imc3"):
            continue
        try:
            ch = read_raw(f)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("Skipped %s: %s", f.name, exc)
            continue
        if ch is None or ch.data.size == 0:
            continue
        # Storage.imcdbc supplies the start time only. The rate is the
        # channel's own x-delta: a rate re-derived as samples / logged duration
        # (as this did until Round 9) is only as right as the sample count, and
        # with the old heuristic reader's count it put the imc3 WFT channels on
        # a 999.93 Hz clock.
        span = spans.get(f.stem)
        if span and span[1] > span[0]:
            ch.start_epoch = span[0]
        channels.append(ch)
    return channels


def _common_timeline(channels) -> tuple[dict, float, dict]:
    """Put every channel on one time axis without touching the WFT samples.

    The time base is the WFT channels' own: ``fs`` is their header rate and
    row n is at ``n / fs``, exactly as FAMOS places sample n of a channel. Its
    length is the longest WFT channel at that rate.

    * A channel at the same rate is placed sample for sample from row 0 (every
      channel of a recording starts at the same instant -- Storage.imcdbc
      gives one start time for all of them). A shorter one is padded with NaN
      at the end; a longer non-WFT one is cut at the end. Nothing is resampled.
    * A channel at a different rate (GPS at 5 Hz) is interpolated from its own
      ``k * x_delta`` axis onto the rows, NaN past its last sample. These are
      never WFT channels; if one were, it would be refused, not interpolated.

    This replaced stretching every channel end to end onto the first file's
    length (``np.interp`` over ``linspace(0, 1, n)``). On the 2006-08-25
    recording the first file was ``Acceleration.raw``, 29 samples longer than
    the WFT channels once both carried their padding, so every WFT channel was
    linearly resampled by 5.22 ppm -- the drift FAMOS showed.

    Returns ``(columns, fs, length_adjust)``; ``length_adjust`` maps a channel
    name to how many samples were padded (+) or cut (-) to fit.
    """
    from dtt.preprocessing import is_wft_channel

    wft = [c for c in channels if is_wft_channel(c.name)]
    master = wft[0] if wft else channels[0]
    fs = master.fs

    def same_rate(c):
        return np.isclose(c.fs, fs, rtol=1e-9, atol=0.0)

    for c in wft:
        if not same_rate(c):
            raise ValueError(
                f"WFT channel {c.name} is sampled at {c.fs:g} Hz and {master.name} "
                f"at {fs:g} Hz; WFT channels are never resampled, so they cannot "
                f"share one time axis")
    n = max(c.data.size for c in (wft or [master]))
    t = np.arange(n) / fs

    out = {"Time": t}
    length_adjust: Dict[str, int] = {}
    for c in channels:
        if same_rate(c):
            if c.data.size >= n:
                arr = c.data[:n]
            else:
                arr = np.full(n, np.nan)
                arr[:c.data.size] = c.data
            if c.data.size != n:
                length_adjust[c.name] = n - c.data.size
        else:
            tc = np.arange(c.data.size) / c.fs
            arr = np.interp(t, tc, c.data, left=np.nan, right=np.nan)
        out[c.name] = arr
    if length_adjust:
        logger.info("Channels fitted to the %d-sample WFT timeline "
                    "(+ padded with NaN at the end, - cut at the end): %s",
                    n, ", ".join(f"{k} {v:+d}" for k, v in length_adjust.items()))
    return out, fs, length_adjust


def _assemble(channels, target_fs, source, famos: bool = True,
              deglitch: bool = False, despike: bool = False,
              despike_rail_min_run: int = 3, despike_dropout_max_run: int = 5,
              despike_hw_cutoff_hz: float = 200.0, despike_net: bool = False,
              despike_net_nsigma: float = 6.0, despike_net_window_s: float = 0.011,
              transient_despike: bool = False, transient_despike_pct: float = 20.0,
              transient_despike_window_s: float = 1.0,
              transient_despike_noise_floor_mult: float = 10.0,
              transient_despike_max_spike_frac: float = 0.4
              ) -> tuple[pd.DataFrame, dict]:
    if not channels:
        return pd.DataFrame(), {"error": "no imc channels found"}

    out, fs, length_adjust = _common_timeline(channels)
    df = pd.DataFrame(out)

    fs_out = fs
    applied = {}
    raw_frame = None
    step = max(1, int(round(fs / target_fs))) if (target_fs and fs > target_fs) else 1
    if famos:
        # Keep an unconditioned copy of every channel the recipe conditions,
        # before smo/FiltLP run. Conditioning at the native rate is correct but
        # it is also destructive, and nothing downstream can reconstruct what
        # was removed — so a study ingested this way has no "before" at all
        # unless it is captured here. Decimated with red() alone so it shares
        # the output time base; that is the genuine raw signal at 100 Hz,
        # aliasing and spikes included.
        #
        # This used to keep the Fx/Fy/Fz forces only. `smo(0.1)` runs on the
        # moments too (FR_Mx_2, RR_Mz_1, …), and `smo(0.5)`/`FiltLP` on Latacc,
        # Longacc and Vehicle_Speed — none of which had a raw counterpart, so
        # the Preprocess screen could not draw a before/after for them and fell
        # back to a single trace that is indistinguishable from raw. Ask the
        # recipe which channels it changes and keep raw for exactly those.
        from dtt.preprocessing import apply_famos_recipe, conditions_channel
        raw_cols = [c for c in df.columns if conditions_channel(c)]
        if raw_cols:
            raw_frame = df[raw_cols].iloc[::step].reset_index(drop=True)
            raw_frame.insert(0, "Time", np.arange(len(raw_frame)) / (fs / step))

        # FAMOS order: smo/FiltLP at the native rate, *then* red(). Decimating
        # first would alias the whole 50-500 Hz band back over the signal, which
        # is what produces phantom spikes in a 1000 Hz WFT recording.
        df, fs_out, applied = apply_famos_recipe(
            df, fs, decimate_factor=step, deglitch=deglitch,
            despike_enabled=despike,
            despike_rail_min_run=despike_rail_min_run,
            despike_dropout_max_run=despike_dropout_max_run,
            despike_hw_cutoff_hz=despike_hw_cutoff_hz,
            despike_net=despike_net,
            despike_net_nsigma=despike_net_nsigma,
            despike_net_window_s=despike_net_window_s,
            transient_enabled=transient_despike,
            transient_pct=transient_despike_pct,
            transient_window_s=transient_despike_window_s,
            transient_noise_floor_mult=transient_despike_noise_floor_mult,
            transient_max_spike_frac=transient_despike_max_spike_frac,
            # Store the pre-smoothing FiltLP intermediate (Latacc_LPF). It is
            # one extra column on one channel, and it is the only stage of the
            # recipe that is *not* recoverable after the fact: red(smo(x)) is
            # the final output, so the smo stage costs nothing to store, but
            # FiltLP's output is consumed by smo and then gone. Keeping it is
            # what lets the Preprocess screen show a bit-exact intermediate
            # instead of recomputing an approximation at the decimated rate.
            # A genuine FAMOS CSV export carries this column too.
            emit_lpf_columns=True)
    elif step > 1:
        df = df.iloc[::step].reset_index(drop=True)
        fs_out = fs / step
        df["Time"] = np.arange(len(df)) / fs_out

    # Earliest channel start, from Storage.imcdbc. This is how several sessions
    # of one route get put back in the order they were driven.
    epochs = [c.start_epoch for c in channels
              if getattr(c, "start_epoch", 0) and np.isfinite(c.start_epoch)]
    meta = {
        "source": str(source),
        "start_epoch": min(epochs) if epochs else None,
        "n_channels": len(channels),
        "raw_fs_hz": round(fs, 3),
        "output_fs_hz": round(fs_out, 3),
        "rows": len(df),
        "duration_s": round(len(df) / fs_out, 2) if fs_out else 0,
        "units": {c.name: c.unit for c in channels},
        "factors": {c.name: c.factor for c in channels},
        "famos_recipe": applied,
        "famos_decimate": step,
        "deglitch": bool(deglitch),
        "despike": bool(despike),
        "raw_frame": raw_frame,
        "length_adjust": length_adjust,
    }
    return df, meta


def read_folder(folder: Path, target_fs: Optional[float] = None,
                famos: bool = True, deglitch: bool = False,
                despike: bool = False, despike_rail_min_run: int = 3,
                despike_dropout_max_run: int = 5, despike_hw_cutoff_hz: float = 200.0,
                despike_net: bool = False, despike_net_nsigma: float = 6.0,
                despike_net_window_s: float = 0.011,
                transient_despike: bool = False, transient_despike_pct: float = 20.0,
                transient_despike_window_s: float = 1.0,
                transient_despike_noise_floor_mult: float = 10.0,
                transient_despike_max_spike_frac: float = 0.4
                ) -> tuple[pd.DataFrame, dict]:
    """Read every channel in an imc raw folder into one time-aligned DataFrame.

    With ``famos`` (default) the imc/FAMOS recipe — ``smo`` / ``FiltLP`` at the
    native rate followed by ``red()`` — is applied as the data is assembled, so
    the frame matches a FAMOS export rather than a raw stride-decimation.
    ``despike`` (off by default) runs the physical-rule despike ahead of it;
    see :func:`dtt.preprocessing.despike`. ``transient_despike`` (off by
    default) runs the manual's "20% within 1s" rule after conditioning; see
    :func:`dtt.preprocessing.detect_transient_spikes`.
    """
    folder = resolve_raw_folder(folder)
    spans = read_imcdbc(folder)
    files = sorted(folder.glob("*.raw"))
    return _assemble(_read_channels(files, spans), target_fs, folder,
                     famos=famos, deglitch=deglitch, despike=despike,
                     despike_rail_min_run=despike_rail_min_run,
                     despike_dropout_max_run=despike_dropout_max_run,
                     despike_hw_cutoff_hz=despike_hw_cutoff_hz,
                     despike_net=despike_net, despike_net_nsigma=despike_net_nsigma,
                     despike_net_window_s=despike_net_window_s,
                     transient_despike=transient_despike,
                     transient_despike_pct=transient_despike_pct,
                     transient_despike_window_s=transient_despike_window_s,
                     transient_despike_noise_floor_mult=transient_despike_noise_floor_mult,
                     transient_despike_max_spike_frac=transient_despike_max_spike_frac)


def read_files(files, target_fs: Optional[float] = None,
               famos: bool = True, deglitch: bool = False,
               despike: bool = False, despike_rail_min_run: int = 3,
               despike_dropout_max_run: int = 5, despike_hw_cutoff_hz: float = 200.0,
               despike_net: bool = False, despike_net_nsigma: float = 6.0,
               despike_net_window_s: float = 0.011,
               transient_despike: bool = False, transient_despike_pct: float = 20.0,
               transient_despike_window_s: float = 1.0,
               transient_despike_noise_floor_mult: float = 10.0,
               transient_despike_max_spike_frac: float = 0.4
               ) -> tuple[pd.DataFrame, dict]:
    """Read a specific list of imc ``.raw`` files (a subset of a recording)."""
    files = [Path(f) for f in files]
    spans = read_imcdbc(files[0].parent) if files else {}
    return _assemble(_read_channels(files, spans), target_fs, "selected files",
                     famos=famos, deglitch=deglitch, despike=despike,
                     despike_rail_min_run=despike_rail_min_run,
                     despike_dropout_max_run=despike_dropout_max_run,
                     despike_hw_cutoff_hz=despike_hw_cutoff_hz,
                     despike_net=despike_net, despike_net_nsigma=despike_net_nsigma,
                     despike_net_window_s=despike_net_window_s,
                     transient_despike=transient_despike,
                     transient_despike_pct=transient_despike_pct,
                     transient_despike_window_s=transient_despike_window_s,
                     transient_despike_noise_floor_mult=transient_despike_noise_floor_mult,
                     transient_despike_max_spike_frac=transient_despike_max_spike_frac)