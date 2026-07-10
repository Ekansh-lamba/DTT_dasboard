"""
imc STUDIO raw-format reader (FAMOS-grade signal + calibration).

Each imc `.raw` file stores ONE channel as a header of ``|XXn`` key blocks
followed by a flat sample blob, then a small trailer (``|CS|CJ|CE``).

Decoded per channel:
  * name / comment     from the ``|CN`` block
  * calibration factor + physical unit  from the ``|CM`` block
  * data type          int16 for WFT force/moment channels; 32-bit for GPS
                       position channels (detected from blob size vs rate)
  * samples            the flat blob between the header and the ``|CS`` trailer

Physical value = raw_sample * factor   (unit from the CM block, e.g. "N").
The per-channel factor was validated against the reference FAMOS CSV export
(e.g. FL_Fz median ≈ 594 daN vs 608 daN in the CSV).

A companion ``Storage.imcdbc`` (XML) gives each channel's start/end epoch,
used for the true sample rate and for time-synchronising channels.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# raw imc channel name  ->  pipeline channel name  (from the FAMOS mapping file)
#   WFT_Fx_fl -> FL_Fx , etc.
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
    """CM block: <u32><double factor> ... <len-prefixed unit string>."""
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


def _smoothness(a: np.ndarray) -> float:
    if a.size < 4:
        return 1.0
    a = a - a.mean()
    return float(np.std(np.diff(a)) / (np.std(a) + 1e-9))


def _locate_data(raw: bytes, itemsize: int) -> int:
    """Find the first byte of the sample blob: the header ends where the
    byte stream stops looking like ascii metadata and becomes a smooth signal.
    """
    cp = _find_block(raw, b"|CP1")
    cs = raw.find(b"|CS")
    if cs < 0:
        cs = len(raw)
    dt = "<i2" if itemsize == 2 else "<i4"
    best_s, best_r = cp + 4, 1e9
    for s in range(cp + 4, min(cp + 2000, cs - 40000), 2):
        seg = np.frombuffer(raw[s:s + 40000], dtype=dt).astype(float)
        r = _smoothness(seg)
        if r < best_r:
            best_r, best_s = r, s
    return best_s


def read_raw(path: Path, fs: float, itemsize: int = 2,
             n_force: Optional[int] = None) -> Optional[ImcChannel]:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|imc3"):
        return None
    cs = raw.find(b"|CS")
    if cs < 0:
        cs = len(raw)
    raw_name, comment = _channel_name(raw)
    factor, unit = _cm_factor_unit(raw)
    dt = "<i2" if itemsize == 2 else "<i4"
    # Data blob ends at the trailer; its length (sample count) is identical for
    # all same-rate channels in a recording, so a shared count anchors the
    # start deterministically:  start = cs - n * itemsize.
    if n_force is not None:
        n = n_force
        start = cs - n * itemsize
    else:
        start = _locate_data(raw, itemsize)
        n = (cs - start) // itemsize
    if start < 0:
        start = _locate_data(raw, itemsize)
        n = (cs - start) // itemsize
    samples = np.frombuffer(raw[start:start + n * itemsize], dtype=dt).astype(np.float64)
    data = samples * factor

    pipeline_name = _map_name(raw_name)
    return ImcChannel(
        name=pipeline_name, raw_name=raw_name, unit=unit,
        factor=factor, data=data, fs=fs, comment=comment,
    )


def _map_name(raw_name: str) -> str:
    m = _WFT_RE.match(raw_name)
    if m:
        comp, pos = m.group(1), m.group(2).upper()
        comp = comp[0].upper() + comp[1].lower()      # Fx / Mz normalised
        return f"{pos}_{comp}"
    return _AUX_MAP.get(raw_name, raw_name)


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


def read_famos(path: Path) -> Optional[ImcChannel]:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|CF"):
        return None
    dx = None
    factor, offset, unit = 1.0, 0.0, ""
    dtype = "<i2"
    name = Path(path).stem
    data_bytes = b""
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
            try:
                factor, offset = float(fields[1]), float(fields[2])
            except ValueError:
                pass
            unit = fields[-1].decode("latin1").strip()
        elif key == "CN" and len(fields) >= 5:
            name = fields[4].decode("latin1").strip()
        elif key == "CS":
            ci = content.find(b",")
            data_bytes = content[ci + 1:] if ci >= 0 else content

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

    # imc3: every same-rate WFT channel shares one sample count, which anchors
    # the data start deterministically across header-length quirks.
    locked = [
        (raw.find(b"|CS") - _locate_data(raw, 2)) // 2
        for raw in (f.read_bytes() for f in files if _WFT_RE.match(f.stem))
        if raw.find(b"|CS") >= 0
    ]
    ref_n = int(np.median(locked)) if locked else 0
    for f in files:
        if not f.read_bytes()[:8].startswith(b"|imc3"):
            continue
        is_wft = bool(_WFT_RE.match(f.stem))
        itemsize = 4 if (ref_n and not is_wft
                         and f.stat().st_size // 2 > ref_n * 1.6) else 2
        n_force = ref_n if (is_wft and ref_n) else None
        try:
            ch = read_raw(f, fs=1.0, itemsize=itemsize, n_force=n_force)
        except Exception:
            continue
        if ch is None or ch.data.size == 0:
            continue
        span = spans.get(f.stem)
        if span and span[1] > span[0]:
            ch.fs = ch.data.size / (span[1] - span[0])
            ch.start_epoch = span[0]
        channels.append(ch)
    return channels


def _assemble(channels, target_fs, source) -> tuple[pd.DataFrame, dict]:
    if not channels:
        return pd.DataFrame(), {"error": "no imc channels found"}

    wft = [c for c in channels if _WFT_RE.match(c.raw_name)]
    master = wft[0] if wft else channels[0]
    fs = master.fs
    n = master.data.size

    out = {"Time": np.arange(n) / fs}
    for c in channels:
        if c.data.size == n:
            arr = c.data
        else:
            xp = np.linspace(0, 1, c.data.size)
            arr = np.interp(np.linspace(0, 1, n), xp, c.data)
        out[c.name] = arr
    df = pd.DataFrame(out)

    if target_fs and fs > target_fs:
        step = max(1, int(round(fs / target_fs)))
        df = df.iloc[::step].reset_index(drop=True)
        fs_out = fs / step
        df["Time"] = np.arange(len(df)) / fs_out
    else:
        fs_out = fs

    meta = {
        "source": str(source),
        "n_channels": len(channels),
        "raw_fs_hz": round(fs, 3),
        "output_fs_hz": round(fs_out, 3),
        "rows": len(df),
        "duration_s": round(len(df) / fs_out, 2) if fs_out else 0,
        "units": {c.name: c.unit for c in channels},
        "factors": {c.name: c.factor for c in channels},
    }
    return df, meta


def read_folder(folder: Path, target_fs: Optional[float] = None) -> tuple[pd.DataFrame, dict]:
    """Read every channel in an imc raw folder into one time-aligned DataFrame."""
    folder = resolve_raw_folder(folder)
    spans = read_imcdbc(folder)
    files = sorted(folder.glob("*.raw"))
    return _assemble(_read_channels(files, spans), target_fs, folder)


def read_files(files, target_fs: Optional[float] = None) -> tuple[pd.DataFrame, dict]:
    """Read a specific list of imc ``.raw`` files (a subset of a recording)."""
    files = [Path(f) for f in files]
    spans = read_imcdbc(files[0].parent) if files else {}
    return _assemble(_read_channels(files, spans), target_fs, "selected files")