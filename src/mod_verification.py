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


def verify_multiple_storm(
    storms: List[str],
    threshold: float,
    metric: str,
    models: List[str],
    hh: int,
    lead_time: int,
    sample_mode: str = "strict",
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[float],
    List[int],
]:
    """Verify and aggregate one threshold period from multiple storms.

    Each entry in ``storms`` must be an A-deck filename such as
    ``"aep172026.dat"``. The basin/file prefixes currently supported by
    :func:`verify_single_storm` are ``aal``, ``awp``, and ``aep``.

    For every storm, :func:`mod_abdeck.find_bdeck_threshold_period` selects
    the first contiguous B-deck interval in which ``metric`` is strictly
    greater than ``threshold`` (strictly less for PMIN). Its exclusive end
    is converted to the last qualifying B-deck cycle before calling
    :func:`verify_single_storm`, whose end-cycle argument is inclusive.
    Storms that never meet the threshold are skipped.

    The mean absolute and signed errors pool all finite cases from all
    selected storms at each model and lead time. Thus storms contribute in
    proportion to their number of verified cycles. Correlation is first
    calculated by :func:`verify_single_storm` for each storm/model and then
    averaged with equal weight over storms having a finite correlation.

    Parameters
    ----------
    storms : list[str]
        A-deck filenames of the form ``a{basin}{storm_number}{yyyy}.dat``,
        for example ``["aep172026.dat", "aal052026.dat"]``.
    threshold : float
        B-deck threshold used to select each storm's verification period.
    metric : str
        One of VMAX, PMIN, RMW, R34, R50, or R64.
    models : list[str]
        Four-character ATCF technique identifiers.
    hh : int
        Verification interval in hours.
    lead_time : int
        Verification-window length. Leads are ``0`` through
        ``(int(lead_time / hh) - 1) * hh``.
    sample_mode : {"strict", "relaxed"}, default="strict"
        Sampling rule passed unchanged to :func:`verify_single_storm`.

    Returns
    -------
    errs_abs : list[list[float]]
        Pooled mean absolute error at each lead for each model.
    errs_bias : list[list[float]]
        Pooled mean signed error at each lead for each model.
    corr : list[float]
        Mean of the finite per-storm correlations for each model.
    n_cases : list[int]
        Total verified cases across storms at each lead time.
    """
    if not isinstance(storms, list) or not storms:
        raise ValueError("storms must be a non-empty list of A-deck filenames.")

    normalized_storms = [str(storm).strip().lower() for storm in storms]
    if len(set(normalized_storms)) != len(normalized_storms):
        raise ValueError("storms must not contain duplicates.")

    parsed_storms = []
    basin_ids = {"al": "L", "wp": "W", "ep": "E"}
    for storm in normalized_storms:
        match = re.fullmatch(r"a(al|wp|ep)(\d{2})(\d{4})\.dat", storm)
        if match is None:
            raise ValueError(
                "Every storm must be an A-deck filename such as "
                "'aal052026.dat', 'awp012026.dat', or 'aep172026.dat'."
            )
        adeck_path = ADECK_DIR / storm
        if not adeck_path.is_file():
            raise FileNotFoundError(f"A-deck file not found: {adeck_path}")
        basin_code, storm_number, year = match.groups()
        parsed_storms.append((storm_number + basin_ids[basin_code], year))

    metric = str(metric).strip().upper()
    sample_mode = str(sample_mode).strip().lower()
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
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    normalized_models = [str(model).strip().upper() for model in models]
    if any(
        not re.fullmatch(r"[A-Z0-9]{4}", model)
        for model in normalized_models
    ):
        raise ValueError("Every model must be a four-character ATCF identifier.")
    if len(set(normalized_models)) != len(normalized_models):
        raise ValueError("models must not contain duplicates.")
    if isinstance(hh, bool) or not isinstance(hh, int) or hh <= 0:
        raise ValueError("hh must be a positive integer.")
    if (
        isinstance(lead_time, bool)
        or not isinstance(lead_time, int)
        or lead_time < hh
    ):
        raise ValueError("lead_time must be an integer greater than or equal to hh.")
    if sample_mode not in {"strict", "relaxed"}:
        raise ValueError("sample_mode must be either 'strict' or 'relaxed'.")

    n_models = len(normalized_models)
    n_leads = int(lead_time / hh)
    absolute_error_sums = np.zeros((n_models, n_leads), dtype=float)
    signed_error_sums = np.zeros((n_models, n_leads), dtype=float)
    correlation_sums = np.zeros(n_models, dtype=float)
    correlation_counts = np.zeros(n_models, dtype=int)
    n_cases = np.zeros(n_leads, dtype=int)

    for storm_id, year in parsed_storms:
        start_cycle, exclusive_end_cycle = abdeck.find_bdeck_threshold_period(
            storm_id=storm_id,
            year=year,
            metric=metric,
            threshold=threshold_value,
            bdeck_dir=BDECK_DIR,
        )
        if start_cycle is None:
            continue

        end_cycle = None
        if exclusive_end_cycle is not None:
            storm_number, basin_id = storm_id[:2], storm_id[2]
            bdeck_path = (
                BDECK_DIR
                / f"b{BASIN_FILE_CODES[basin_id]}{storm_number}{year}.dat"
            )
            start_dt = abdeck._parse_cycle(start_cycle, "start_cycle")
            exclusive_end_dt = abdeck._parse_cycle(
                exclusive_end_cycle, "end_cycle"
            )
            qualifying_times = [
                cycle
                for cycle in abdeck._read_bdeck_times(bdeck_path)
                if start_dt <= cycle < exclusive_end_dt
            ]
            if not qualifying_times:
                continue
            end_cycle = max(qualifying_times).strftime("%Y%m%d%H")

        storm_errs, _, _, storm_corr, storm_n_cases = verify_single_storm(
            storm_id=storm_id,
            year=year,
            start_cycle=start_cycle,
            end_cycle=end_cycle,
            models=normalized_models,
            hh=hh,
            lead_time=lead_time,
            metric=metric,
            sample_mode=sample_mode,
        )

        n_cases += np.asarray(storm_n_cases, dtype=int)
        for model_index, model_errors in enumerate(storm_errs):
            for lead_index in range(n_leads):
                valid_errors = model_errors[:, lead_index]
                valid_errors = valid_errors[np.isfinite(valid_errors)]
                if valid_errors.size != storm_n_cases[lead_index]:
                    raise RuntimeError(
                        "Internal sample-count mismatch while aggregating storms."
                    )
                absolute_error_sums[model_index, lead_index] += np.sum(
                    np.abs(valid_errors)
                )
                signed_error_sums[model_index, lead_index] += np.sum(valid_errors)

            if np.isfinite(storm_corr[model_index]):
                correlation_sums[model_index] += storm_corr[model_index]
                correlation_counts[model_index] += 1

    errs_abs = np.full((n_models, n_leads), np.nan, dtype=float)
    errs_bias = np.full((n_models, n_leads), np.nan, dtype=float)
    nonzero_leads = n_cases > 0
    errs_abs[:, nonzero_leads] = (
        absolute_error_sums[:, nonzero_leads] / n_cases[nonzero_leads]
    )
    errs_bias[:, nonzero_leads] = (
        signed_error_sums[:, nonzero_leads] / n_cases[nonzero_leads]
    )

    corr = np.full(n_models, np.nan, dtype=float)
    valid_correlations = correlation_counts > 0
    corr[valid_correlations] = (
        correlation_sums[valid_correlations]
        / correlation_counts[valid_correlations]
    )

    return errs_abs.tolist(), errs_bias.tolist(), corr.tolist(), n_cases.tolist()
