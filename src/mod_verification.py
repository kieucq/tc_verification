from __future__ import annotations

import re
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np

if __package__:
    from . import mod_abdeck as abdeck
else:
    import mod_abdeck as abdeck


PROJECT_ROOT = abdeck.PROJECT_ROOT
ADECK_DIR = abdeck.ADECK_DIR
BDECK_DIR = abdeck.BDECK_DIR
BASIN_FILE_CODES = abdeck.BASIN_FILE_CODES
SUPPORTED_METRICS = abdeck.SUPPORTED_METRICS


def verify_single_storm(
    storm_id: str,
    year: str,
    start_cycle: str,
    end_cycle: Optional[str] = None,
    models: Optional[List[str]] = None,
    hh: Optional[int] = None,
    lead_time: Optional[int] = None,
    metric: Optional[str] = None,
    sample_mode: str = "strict",
) -> Tuple[
    List[np.ndarray],
    List[List[float]],
    List[List[float]],
    List[float],
    List[int],
]:
    """Verify one metric for multiple models on a homogeneous sample.

    Parameters
    ----------
    storm_id : str
        Two-digit storm number followed by L, W, or E, for example ``'17E'``.
    year : str
        Four-digit year.
    start_cycle : str
        Inclusive lower initialization-time bound in ``yyyymmddhh`` format.
    end_cycle : str or None, optional
        Inclusive upper initialization-time bound in ``yyyymmddhh`` format.
        If omitted, ``None``, or blank, all common initialization cycles on
        or after ``start_cycle`` are considered.
    models : list[str]
        Four-character ATCF technique identifiers.
    hh : int
        Verification interval in hours.
    lead_time : int
        Verification-window length. Leads are ``0`` through
        ``(int(lead_time / hh) - 1) * hh``.
    metric : str
        One of VMAX, PMIN, RMW, R34, R50, or R64.
    sample_mode : str, default="strict"
        ``"strict"`` requires a B-deck timestamp at ``cycle + lead_time``
        and complete data for every requested model and verified lead. The
        same cycles are used at every lead. ``"relaxed"`` selects complete
        model/best-track cases independently at each lead, so sample sizes
        can vary with lead time.

    Returns
    -------
    errs : list[np.ndarray]
        One N-by-T forecast-minus-best-track error array per model. In
        relaxed mode, N is the union of cycles verified at one or more leads;
        unavailable cycle/lead cells contain NaN.
    errs_abs : list[list[float]]
        Mean absolute error at each lead for each model, using the applicable
        per-lead sample.
    errs_bias : list[list[float]]
        Mean signed error at each lead for each model, using the applicable
        per-lead sample.
    corr : list[float]
        Pearson correlation over all N-by-T forecast/best-track pairs for
        each model. In relaxed mode, missing cells are excluded. Returns NaN
        when correlation is undefined.
    n_cases : list[int]
        Number of verified cases at each lead. Counts are identical in strict
        mode and may vary by lead in relaxed mode.
    """
    storm_id = str(storm_id).strip().upper()
    year = str(year).strip()
    metric = str(metric).strip().upper()
    sample_mode = str(sample_mode).strip().lower()

    if not re.fullmatch(r"\d{2}[LWE]", storm_id):
        raise ValueError("storm_id must be two digits followed by L, W, or E.")
    if not re.fullmatch(r"\d{4}", year):
        raise ValueError("year must be a four-digit string.")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")

    normalized_models = [str(model).strip().upper() for model in models]
    if any(not re.fullmatch(r"[A-Z0-9]{4}", model) for model in normalized_models):
        raise ValueError("Every model must be a four-character ATCF identifier.")
    if len(set(normalized_models)) != len(normalized_models):
        raise ValueError("models must not contain duplicates.")
    if isinstance(hh, bool) or not isinstance(hh, int) or hh <= 0:
        raise ValueError("hh must be a positive integer.")
    if isinstance(lead_time, bool) or not isinstance(lead_time, int) or lead_time < hh:
        raise ValueError("lead_time must be an integer greater than or equal to hh.")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}.")
    if sample_mode not in {"strict", "relaxed"}:
        raise ValueError("sample_mode must be either 'strict' or 'relaxed'.")

    start_dt = abdeck._parse_cycle(start_cycle, "start_cycle")
    end_cycle_missing = end_cycle is None or (
        isinstance(end_cycle, str) and not end_cycle.strip()
    )
    end_dt = (
        None
        if end_cycle_missing
        else abdeck._parse_cycle(end_cycle, "end_cycle")
    )
    if end_dt is not None and start_dt > end_dt:
        raise ValueError("start_cycle must not be later than end_cycle.")

    storm_number, basin_id = storm_id[:2], storm_id[2]
    basin_code = BASIN_FILE_CODES[basin_id]
    adeck_path = ADECK_DIR / f"a{basin_code}{storm_number}{year}.dat"
    bdeck_path = BDECK_DIR / f"b{basin_code}{storm_number}{year}.dat"
    if not adeck_path.is_file():
        raise FileNotFoundError(f"A-deck file not found: {adeck_path}")
    if not bdeck_path.is_file():
        raise FileNotFoundError(f"B-deck file not found: {bdeck_path}")

    forecast = abdeck._read_adeck_metric(
        adeck_path, metric, set(normalized_models)
    )
    best_track = abdeck._read_bdeck_metric(bdeck_path, metric)
    n_leads = int(lead_time / hh)
    lead_hours = np.arange(n_leads, dtype=int) * hh

    cycles_by_model = []
    for model in normalized_models:
        cycles_by_model.append(
            {cycle for cycle, record_model, _ in forecast if record_model == model}
        )
    candidate_cycles = set.intersection(*cycles_by_model)
    candidate_cycles = {
        cycle
        for cycle in candidate_cycles
        if cycle >= start_dt and (end_dt is None or cycle <= end_dt)
    }

    def has_complete_case(cycle, tau: int) -> bool:
        return (
            cycle + timedelta(hours=tau) in best_track
            and all(
                (cycle, model, tau) in forecast
                for model in normalized_models
            )
        )

    sorted_candidates = sorted(candidate_cycles)
    if sample_mode == "strict":
        bdeck_times = abdeck._read_bdeck_times(bdeck_path)
        homogeneous_cycles = [
            cycle
            for cycle in sorted_candidates
            if cycle + timedelta(hours=lead_time) in bdeck_times
            and all(
                has_complete_case(cycle, int(tau))
                for tau in lead_hours
            )
        ]
        cycles_by_lead = [homogeneous_cycles for _ in lead_hours]
    else:
        cycles_by_lead = [
            [
                cycle
                for cycle in sorted_candidates
                if has_complete_case(cycle, int(tau))
            ]
            for tau in lead_hours
        ]
        homogeneous_cycles = sorted(
            {cycle for cycles in cycles_by_lead for cycle in cycles}
        )

    n_cases = [len(cycles) for cycles in cycles_by_lead]
    n_cycles = len(homogeneous_cycles)
    cycle_rows = {cycle: row for row, cycle in enumerate(homogeneous_cycles)}
    observed = np.full((n_cycles, n_leads), np.nan, dtype=float)
    forecast_by_model = {
        model: np.full((n_cycles, n_leads), np.nan, dtype=float)
        for model in normalized_models
    }

    for column, (tau, valid_cycles) in enumerate(zip(lead_hours, cycles_by_lead)):
        tau = int(tau)
        for cycle in valid_cycles:
            row = cycle_rows[cycle]
            observed[row, column] = best_track[
                cycle + timedelta(hours=tau)
            ]
            for model in normalized_models:
                forecast_by_model[model][row, column] = forecast[
                    (cycle, model, tau)
                ]

    errs: List[np.ndarray] = []
    errs_abs: List[List[float]] = []
    errs_bias: List[List[float]] = []
    corr: List[float] = []

    for model in normalized_models:
        forecast_values = forecast_by_model[model]
        model_errors = forecast_values - observed
        errs.append(model_errors)

        model_mae = []
        model_bias = []
        for column, count in enumerate(n_cases):
            if count:
                column_errors = model_errors[:, column]
                valid_errors = column_errors[np.isfinite(column_errors)]
                model_mae.append(float(np.mean(np.abs(valid_errors))))
                model_bias.append(float(np.mean(valid_errors)))
            else:
                model_mae.append(float("nan"))
                model_bias.append(float("nan"))
        errs_abs.append(model_mae)
        errs_bias.append(model_bias)
        corr.append(abdeck._pearson_correlation(forecast_values, observed))

    return errs, errs_abs, errs_bias, corr, n_cases
