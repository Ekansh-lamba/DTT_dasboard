from __future__ import annotations
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
_EVENT_SIG = b"|RC5"
_EVENT_LEN = 16
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


def _strip_event_records(blob: bytes) -> bytes:
    """Remove the embedded 16-byte ``|RC5`` event records from a data blob.

    imc STUDIO interleaves these marker records into the streamed samples; read
    as int16 they show up as ~40k spikes per channel. FAMOS skips them.
    """
    if _EVENT_SIG not in blob:
        return blob
    out = bytearray()
    i = blob.find(_EVENT_SIG)
    prev = 0
    while i != -1:
        out += blob[prev:i]
        prev = i + _EVENT_LEN
        i = blob.find(_EVENT_SIG, prev)
    out += blob[prev:]
    return bytes(out)


def _find_data_start(clean: bytes, itemsize: int = 2) -> int:
    """Byte offset in the marker-stripped blob where the ascii header ends and
    smooth signal begins. Picks the offset whose window is both smoothest and
    not full-scale garbage (which is how a wrong byte alignment looks).
    """
    dt = "<i2" if itemsize == 2 else "<i4"
    best_s, best = 0, 1e18
    hi = max(1, min(600, len(clean) - 40000))
    for s in range(0, hi):
        seg = np.frombuffer(clean[s:s + 40000], dtype=dt).astype(float)
        seg = seg - seg.mean()
        smooth = np.std(np.diff(seg)) / (np.std(seg) + 1e-9)   # low = smooth
        rng = np.percentile(np.abs(seg), 99) / 32768.0          # ~1 if garbage
        if smooth + rng < best:
            best, best_s = smooth + rng, s
    return best_s


def read_raw(path: Path, fs: float, itemsize: int = 2,
             n_force: Optional[int] = None) -> Optional[ImcChannel]:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"|imc3"):
        return None
    cs = raw.rfind(b"|CS")               # trailer (last occurrence)
    if cs < 0:
        cs = len(raw)
    raw_name, comment = _channel_name(raw)
    factor, unit = _cm_factor_unit(raw)
    cp = raw.find(b"|CP1")
    region = raw[(cp if cp >= 0 else 0):cs]
    clean = _strip_event_records(region)          # drop |RC5 event records
    start = _find_data_start(clean, itemsize)     # header -> data boundary
    end = len(clean) - ((len(clean) - start) % itemsize)
    dt = "<i2" if itemsize == 2 else "<i4"
    samples = np.frombuffer(clean[start:end], dtype=dt).astype(np.float64)
    # Trim the leading header/preamble garbage: _find_data_start's coarse 40k
    # window can land a few hundred bytes early, inside a messy transition where
    # header bytes are interspersed with the first samples (shows as a spike at
    # t=0). Advance to where a stable clean run of real samples begins.
    if samples.size > 3000:
        ref = samples[500:2500]
        m = np.median(ref)
        sd = 1.4826 * np.median(np.abs(ref - m)) or 1.0
        thr = 15.0 * sd
        win = 50
        lim = min(500, samples.size - win)
        k = 0
        while k < lim and not np.all(np.abs(samples[k:k + win] - m) < thr):
            k += 1
        if k:
            samples = samples[k:]
        # trailing garbage (mirror of the leading trim)
        j = samples.size
        low = max(win, samples.size - 500)
        while j > low and not np.all(np.abs(samples[j - win:j] - m) < thr):
            j -= 1
        if j < samples.size:
            samples = samples[:j]
    data = samples * factor

    return ImcChannel(
        name=_map_name(raw_name), raw_name=raw_name, unit=unit,
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
    prelim = []
    wft_counts: List[int] = []
    for f in files:
        if not f.read_bytes()[:8].startswith(b"|imc3"):
            continue
        is_wft = bool(_WFT_RE.match(f.stem))
        try:
            ch = read_raw(f, fs=1.0, itemsize=2)
        except Exception:
            continue
        if ch is None or ch.data.size == 0:
            continue
        if is_wft:
            wft_counts.append(ch.data.size)
        prelim.append((f, is_wft, ch))

    ref_n = int(np.median(wft_counts)) if wft_counts else 0
    for f, is_wft, ch in prelim:
        # A 32-bit aux channel (GPS) read as int16 yields ~2x samples: re-read.
        if not is_wft and ref_n and ch.data.size > ref_n * 1.6:
            try:
                ch32 = read_raw(f, fs=1.0, itemsize=4)
                if ch32 is not None and ch32.data.size:
                    ch = ch32
            except Exception:
                pass
        span = spans.get(f.stem)
        if span and span[1] > span[0]:
            ch.fs = ch.data.size / (span[1] - span[0])   # -> true ~1000 Hz
            ch.start_epoch = span[0]
        channels.append(ch)
    return channels


def _assemble(channels, target_fs, source, famos: bool = True,
              deglitch: bool = False) -> tuple[pd.DataFrame, dict]:
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

    fs_out = fs
    applied = {}
    step = max(1, int(round(fs / target_fs))) if (target_fs and fs > target_fs) else 1
    if famos:
        # FAMOS order: smo/FiltLP at the native rate, *then* red(). Decimating
        # first would alias the whole 50-500 Hz band back over the signal, which
        # is what produces phantom spikes in a 1000 Hz WFT recording.
        from dtt.preprocessing import apply_famos_recipe
        df, fs_out, applied = apply_famos_recipe(
            df, fs, decimate_factor=step, deglitch=deglitch,
            emit_lpf_columns=False)
    elif step > 1:
        df = df.iloc[::step].reset_index(drop=True)
        fs_out = fs / step
        df["Time"] = np.arange(len(df)) / fs_out

    meta = {
        "source": str(source),
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
    }
    return df, meta


def read_folder(folder: Path, target_fs: Optional[float] = None,
                famos: bool = True, deglitch: bool = False
                ) -> tuple[pd.DataFrame, dict]:
    """Read every channel in an imc raw folder into one time-aligned DataFrame.

    With ``famos`` (default) the imc/FAMOS recipe — ``smo`` / ``FiltLP`` at the
    native rate followed by ``red()`` — is applied as the data is assembled, so
    the frame matches a FAMOS export rather than a raw stride-decimation.
    """
    folder = resolve_raw_folder(folder)
    spans = read_imcdbc(folder)
    files = sorted(folder.glob("*.raw"))
    return _assemble(_read_channels(files, spans), target_fs, folder,
                     famos=famos, deglitch=deglitch)


def read_files(files, target_fs: Optional[float] = None,
               famos: bool = True, deglitch: bool = False
               ) -> tuple[pd.DataFrame, dict]:
    """Read a specific list of imc ``.raw`` files (a subset of a recording)."""
    files = [Path(f) for f in files]
    spans = read_imcdbc(files[0].parent) if files else {}
    return _assemble(_read_channels(files, spans), target_fs, "selected files",
                     famos=famos, deglitch=deglitch)