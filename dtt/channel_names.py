"""One naming convention for channels, applied everywhere a name is shown.

The same wheel channel arrives under different spellings -- ``FR_Fx_2`` in the
imc export, ``FR_Fx`` in a CSV, ``A1R_Fx`` in a truck layout -- and figures,
tables and the report used to print whichever spelling the file happened to
carry. Here:

* :func:`canonical_name` resolves any spelling of a wheel channel to one name:
  ``<position>_<component>``, with FL/FR/RL/RR for a two-axle car position and
  the axle id (``A3LO``) otherwise. ``FR_Fx_2`` and ``FR_Fx`` both become
  ``FR_Fx``. Non-wheel channels keep their own name.
* Optional display names -- ``{"FR_Fx": "Front Right Fx"}`` -- are keyed by
  that canonical name, so one entry covers every export of the channel. They
  live in ``channel_names.json`` in the data folder and are edited on the
  GUI's Channel Names screen.
* :func:`display_name` is what every figure title, legend, table and report
  slide calls. Data files (``processed_data.csv``) and figure *file names*
  keep the source spelling -- renaming those would break every study already
  on disk and the FAMOS cross-check, which matches columns by source name.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

from dtt.channels import parse_channel
from dtt.config import DATA_DIR

logger = logging.getLogger(__name__)

NAMES_FILE = DATA_DIR / "channel_names.json"

_cache: Optional[Dict[str, str]] = None
_cache_mtime: Optional[float] = None


def canonical_name(channel: str) -> str:
    """``FR_Fx_2`` -> ``FR_Fx``; ``A3LO_Fz`` stays; ``Latacc`` stays."""
    parsed = parse_channel(str(channel))
    if parsed is None:
        return str(channel)
    pos, comp = parsed
    return f"{pos.legacy_label or pos.id}_{comp}"


def load_names(path: Path = NAMES_FILE) -> Dict[str, str]:
    """User display names, keyed by canonical name. Missing file -> ``{}``.

    Re-read whenever the file changes, so an edit on the Channel Names screen
    reaches the next figure without a restart.
    """
    global _cache, _cache_mtime
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if path == NAMES_FILE and _cache is not None and mtime == _cache_mtime:
        return dict(_cache)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable channel names file %s: %s", path, exc)
        return {}
    names = {str(k): str(v).strip() for k, v in (raw or {}).items()
             if isinstance(v, str) and v.strip()}
    if path == NAMES_FILE:
        _cache, _cache_mtime = dict(names), mtime
    return names


def save_names(names: Mapping[str, str], path: Path = NAMES_FILE) -> Path:
    """Write display names, dropping blanks and entries equal to the default."""
    clean = {canonical_name(k): v.strip() for k, v in names.items()
             if v and v.strip() and v.strip() != canonical_name(k)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(clean.items())), indent=2),
                    encoding="utf-8")
    global _cache
    _cache = None
    return path


def display_name(channel: str, names: Optional[Mapping[str, str]] = None) -> str:
    """The name to show for ``channel``: user's name, else the canonical one."""
    canon = canonical_name(channel)
    table = load_names() if names is None else names
    return table.get(canon) or table.get(str(channel)) or canon


_MOMENT_RE = re.compile(r"(?:^|[_\W])M[xyz](?:$|[_\W])", re.I)


def channel_unit(channel: str) -> str:
    """Axis unit for ``channel``, following the imc config file's YUNIT lines.

    The Preprocess screen used to hard-code "daN" because the channel list only ever held
    forces. It now lists every column in the processed CSV, so a hard-coded
    force unit would label Latacc in daN and Vehicle_Speed in daN — a plot that
    states the wrong unit is worse than one that states none.
    """
    key = channel.strip().lower()
    if key in ("latacc", "lat_acc", "longacc", "long_acc", "latacc_lpf",
               "acceleration", "accelz"):
        return "m/s²"
    if key in ("velforward", "vellateral"):
        return "m/s"                       # velocities, not accelerations
    if key in ("vehicle_speed", "speed_kmph", "speed2d", "gps.speed"):
        return "km/h"
    if key in ("yawrate", "angratex", "angratey") or "anglespeed" in key:
        return "°/s"
    if key.startswith("angle") or key.endswith("_angle") or "_angle_" in key:
        return "°"
    if key in ("distance", "dist", "altitude"):
        return "m"
    if key in ("latitude", "longitude"):
        return "°"
    from dtt.preprocessing import is_wft_channel
    if is_wft_channel(channel):
        # forces daN, moments daN·m — both smo(0.1) in the recipe
        return "daN·m" if _MOMENT_RE.search(channel) else "daN"
    return ""
