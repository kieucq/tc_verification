from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADECK_DIR = PROJECT_ROOT / "data" / "adeck"
BDECK_DIR = PROJECT_ROOT / "data" / "bdeck"
BASIN_FILE_CODES = {"L": "al", "W": "wp", "E": "ep", "C": "cp"}
SUPPORTED_METRICS = {"VMAX", "PMIN", "RMW", "R34", "R50", "R64"}


def _parse_cycle(value: str, name: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{10}", value):
        raise ValueError(f"{name} must have the form 'yyyymmddhh'.")
    try:
        return datetime.strptime(value, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid date/time: {value!r}") from exc


def _as_int(value: str) -> Optional[int]:
    try:
        return int(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def _positive_field(fields: Sequence[str], index: int) -> Optional[float]:
    if index >= len(fields):
        return None
    value = _as_int(fields[index])
    return float(value) if value is not None and value > 0 else None


def _radius_value(fields: Sequence[str], threshold: int) -> Optional[float]:
    # ATCF fields 12 and 13--16 (one-based) are RAD, wind code, and radii.
    if len(fields) < 17 or _as_int(fields[11]) != threshold:
        return None

    wind_code = fields[12].strip().upper()
    radius_fields = fields[13:17]
    radii = [_as_int(value) for value in radius_fields]

    if wind_code == "AAA":
        radius = radii[0] if radii else None
        return float(radius) if radius is not None and radius >= 0 else None

    # The starting quadrant varies with wind_code, but its mean does not.
    if len(radii) != 4 or any(value is None or value < 0 for value in radii):
        return None
    return float(np.mean(radii))


def _metric_value(fields: Sequence[str], metric: str) -> Optional[float]:
    if metric == "VMAX":
        return _positive_field(fields, 8)
    if metric == "PMIN":
        return _positive_field(fields, 9)
    if metric == "RMW":
        return _positive_field(fields, 19)
    return _radius_value(fields, int(metric[1:]))


def _read_adeck_metric(
    path: Path, metric: str, requested_models: set[str]
) -> Dict[Tuple[datetime, str, int], float]:
    records: Dict[Tuple[datetime, str, int], float] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_fields in csv.reader(handle):
            fields = [field.strip() for field in raw_fields]
            if len(fields) < 10:
                continue
            model = fields[4].upper()
            if model not in requested_models:
                continue
            try:
                cycle = datetime.strptime(fields[2], "%Y%m%d%H")
            except ValueError:
                continue
            tau = _as_int(fields[5])
            value = _metric_value(fields, metric)
            if tau is not None and tau >= 0 and value is not None:
                # VMAX/PMIN/RMW repeat on different RAD lines. Keep one value.
                records.setdefault((cycle, model, tau), value)
    return records


def _read_bdeck_metric(path: Path, metric: str) -> Dict[datetime, float]:
    records: Dict[datetime, float] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_fields in csv.reader(handle):
            fields = [field.strip() for field in raw_fields]
            if len(fields) < 10 or fields[4].upper() != "BEST":
                continue
            try:
                valid_time = datetime.strptime(fields[2], "%Y%m%d%H")
            except ValueError:
                continue
            value = _metric_value(fields, metric)
            if value is not None:
                records.setdefault(valid_time, value)
    return records


def _read_bdeck_times(path: Path) -> set[datetime]:
    """Return every valid BEST-track timestamp present in a B-deck file."""
    valid_times: set[datetime] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_fields in csv.reader(handle):
            fields = [field.strip() for field in raw_fields]
            if len(fields) < 5 or fields[4].upper() != "BEST":
                continue
            try:
                valid_times.add(datetime.strptime(fields[2], "%Y%m%d%H"))
            except ValueError:
                continue
    return valid_times


def find_bdeck_threshold_period(
    storm_id: str,
    year: str,
    metric: str,
    threshold: float,
    *,
    bdeck_dir: Optional[Path | str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Find the first contiguous B-deck period meeting a metric threshold.

    For ``PMIN``, a cycle qualifies when its value is strictly less than the
    threshold. For every other supported metric, a cycle qualifies when its
    value is strictly greater than the threshold.

    The returned interval is ``[start_cycle, end_cycle)``: ``start_cycle`` is
    the first qualifying B-deck cycle, while ``end_cycle`` is the first later
    cycle that is missing the metric or no longer satisfies the strict
    comparison. Consequently, ``end_cycle`` itself is not part of the
    qualifying period. It is ``None`` if the period continues through the
    final B-deck cycle. Both values are ``None`` if no cycle qualifies.

    Parameters
    ----------
    storm_id : str
        Two-digit storm number followed by L, W, E, or C, for example
        ``"17E"``.
    year : str
        Four-digit storm year.
    metric : str
        One of VMAX, PMIN, RMW, R34, R50, or R64.
    threshold : float
        Strict comparison threshold in the metric's native units.
    bdeck_dir : pathlib.Path or str, optional
        Alternate B-deck directory. Defaults to the repository's
        ``data/bdeck`` directory.

    Returns
    -------
    start_cycle, end_cycle : tuple[str | None, str | None]
        Cycle strings in ``yyyymmddhh`` format. ``end_cycle`` is an exclusive
        boundary.
    """
    storm_id = str(storm_id).strip().upper()
    year = str(year).strip()
    metric = str(metric).strip().upper()

    if not re.fullmatch(r"\d{2}[LWEC]", storm_id):
        raise ValueError("storm_id must be two digits followed by L, W, E, or C.")
    if not re.fullmatch(r"\d{4}", year):
        raise ValueError("year must be a four-digit string.")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}.")
    if isinstance(threshold, bool):
        raise ValueError("threshold must be a finite number.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number.") from exc
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be a finite number.")

    storm_number, basin_id = storm_id[:2], storm_id[2]
    basin_code = BASIN_FILE_CODES[basin_id]
    directory = BDECK_DIR if bdeck_dir is None else Path(bdeck_dir)
    bdeck_path = directory / f"b{basin_code}{storm_number}{year}.dat"
    if not bdeck_path.is_file():
        raise FileNotFoundError(f"B-deck file not found: {bdeck_path}")

    metric_values = _read_bdeck_metric(bdeck_path, metric)
    bdeck_times = sorted(_read_bdeck_times(bdeck_path))

    def qualifies(cycle: datetime) -> bool:
        value = metric_values.get(cycle)
        if value is None:
            return False
        if metric == "PMIN":
            return value < threshold_value
        return value > threshold_value

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    for cycle in bdeck_times:
        if start_time is None:
            if qualifies(cycle):
                start_time = cycle
        elif not qualifies(cycle):
            end_time = cycle
            break

    if start_time is None:
        return None, None
    start_cycle = start_time.strftime("%Y%m%d%H")
    end_cycle = end_time.strftime("%Y%m%d%H") if end_time is not None else None
    return start_cycle, end_cycle


def _pearson_correlation(forecast: np.ndarray, observed: np.ndarray) -> float:
    x = np.asarray(forecast, dtype=float).ravel()
    y = np.asarray(observed, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
