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


def compute_ensemble_spread(
    models: List[str],
    metric: str,
    storms: List[str],
    lead_time: int,
    initial_intensity_values: List[float],
) -> Tuple[List[float], List[int]]:
    """Compute ensemble spread by initial-intensity range.

    Every explicit storm must be an A-deck filename such as
    ``"aep172026.dat"``. A storm number of ``ALL``, for example
    ``"aepALL2026.dat"``, expands to every numbered A-deck in that basin
    and year. Basin filename codes ``al``, ``wp``, ``ep``, and ``cp`` are
    supported.

    A forecast cycle is included only when every requested model has the
    selected metric at both lead 0 and ``lead_time``. Its initial intensity
    is the arithmetic mean of the models' lead-0 values. For each requested
    value ``V_i``, cycles with initial ensemble-mean intensity in the closed
    interval ``[V_i - 5, V_i + 5]`` are selected. Ranges are evaluated
    independently, so a cycle can contribute to overlapping ranges.

    Ensemble variance is calculated across models at ``lead_time`` for each
    qualifying cycle using the population definition (``ddof=0``). The
    reported spread is the square root of the mean of those cycle variances
    across all selected storms. This weights every forecast cycle equally.

    Parameters
    ----------
    models : list[str]
        At least two four-character ATCF model identifiers.
    metric : str
        One of VMAX, PMIN, RMW, R34, R50, or R64.
    storms : list[str]
        Explicit A-deck filenames and/or ``ALL`` patterns such as
        ``["aep172026.dat", "aalALL2025.dat"]``.
    lead_time : int
        Non-negative forecast lead time in hours at which spread is
        calculated.
    initial_intensity_values : list[float]
        Centers of the initial-intensity ranges. Each range is ``V_i +/- 5``
        in the selected metric's native units.

    Returns
    -------
    spread : list[float]
        Ensemble spread for each input intensity-range center, in the same
        order as ``initial_intensity_values``. A range with no qualifying
        cycles is NaN.
    n_cases : list[int]
        Number of common forecast cycles in each intensity range.
    """
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("models must be a list containing at least two models.")
    normalized_models = [str(model).strip().upper() for model in models]
    if any(
        not re.fullmatch(r"[A-Z0-9]{4}", model)
        for model in normalized_models
    ):
        raise ValueError("Every model must be a four-character ATCF identifier.")
    if len(set(normalized_models)) != len(normalized_models):
        raise ValueError("models must not contain duplicates.")

    metric = str(metric).strip().upper()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}.")
    if not isinstance(storms, list) or not storms:
        raise ValueError("storms must be a non-empty list of A-deck filenames.")
    normalized_storms = [str(storm).strip().lower() for storm in storms]
    if len(set(normalized_storms)) != len(normalized_storms):
        raise ValueError("storms must not contain duplicates.")
    if (
        isinstance(lead_time, bool)
        or not isinstance(lead_time, int)
        or lead_time < 0
    ):
        raise ValueError("lead_time must be a non-negative integer.")
    if not isinstance(initial_intensity_values, list) or not initial_intensity_values:
        raise ValueError("initial_intensity_values must be a non-empty list.")

    intensity_centers = []
    for value in initial_intensity_values:
        if isinstance(value, bool):
            raise ValueError("Every initial intensity value must be finite.")
        try:
            center = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Every initial intensity value must be finite."
            ) from exc
        if not np.isfinite(center):
            raise ValueError("Every initial intensity value must be finite.")
        intensity_centers.append(center)
    if len(set(intensity_centers)) != len(intensity_centers):
        raise ValueError("initial_intensity_values must not contain duplicates.")

    adeck_paths = []
    seen_paths = set()
    storm_pattern = re.compile(r"a(al|wp|ep|cp)(\d{2}|all)(\d{4})\.dat")
    explicit_pattern = re.compile(r"a(al|wp|ep|cp)\d{2}\d{4}\.dat")
    for storm in normalized_storms:
        match = storm_pattern.fullmatch(storm)
        if match is None:
            raise ValueError(
                "Every storm must look like 'aep172026.dat' or "
                "'aepALL2026.dat'."
            )
        basin_code, storm_number, year = match.groups()
        if storm_number == "all":
            matching_paths = sorted(
                path
                for path in ADECK_DIR.glob(f"a{basin_code}??{year}.dat")
                if path.is_file() and explicit_pattern.fullmatch(path.name)
            )
            if not matching_paths:
                raise FileNotFoundError(
                    "No A-deck files found for "
                    f"basin={basin_code!r}, year={year!r}."
                )
        else:
            path = ADECK_DIR / storm
            if not path.is_file():
                raise FileNotFoundError(f"A-deck file not found: {path}")
            matching_paths = [path]

        for path in matching_paths:
            if path not in seen_paths:
                adeck_paths.append(path)
                seen_paths.add(path)

    variance_sums = np.zeros(len(intensity_centers), dtype=float)
    n_cases = np.zeros(len(intensity_centers), dtype=int)
    requested_models = set(normalized_models)

    for adeck_path in adeck_paths:
        forecast = abdeck._read_adeck_metric(
            adeck_path, metric, requested_models
        )
        cycles_by_model = []
        for model in normalized_models:
            cycles_by_model.append(
                {
                    cycle
                    for cycle, record_model, tau in forecast
                    if record_model == model and tau == lead_time
                }
            )
        common_cycles = set.intersection(*cycles_by_model)
        common_cycles = {
            cycle
            for cycle in common_cycles
            if all(
                (cycle, model, 0) in forecast
                for model in normalized_models
            )
        }

        for cycle in common_cycles:
            initial_values = np.asarray(
                [forecast[(cycle, model, 0)] for model in normalized_models],
                dtype=float,
            )
            forecast_values = np.asarray(
                [
                    forecast[(cycle, model, lead_time)]
                    for model in normalized_models
                ],
                dtype=float,
            )
            initial_intensity = float(np.mean(initial_values))
            ensemble_variance = float(np.var(forecast_values, ddof=0))

            for index, center in enumerate(intensity_centers):
                if center - 5.0 <= initial_intensity <= center + 5.0:
                    variance_sums[index] += ensemble_variance
                    n_cases[index] += 1

    spread = np.full(len(intensity_centers), np.nan, dtype=float)
    has_cases = n_cases > 0
    spread[has_cases] = np.sqrt(
        variance_sums[has_cases] / n_cases[has_cases]
    )
    return spread.tolist(), n_cases.tolist()


def compute_threshold_correlation_statistics(
    storms: List[str],
    models: List[str],
    metric: str,
    threshold_value_1: float,
    threshold_value_2: float,
    threshold_value_3: float,
    threshold_value_4: float,
    hh: int,
    lead_time: int,
    sample_mode: str = "strict",
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[List[int]],
]:
    """Summarize cycle-level intensity-change correlations in two ranges.

    Initialization cycles are assigned using the B-deck VMAX at the cycle:
    the first group contains cycles with
    ``threshold_value_1 < VMAX < threshold_value_2`` and the second contains
    cycles with ``threshold_value_3 < VMAX < threshold_value_4``. All bounds
    are exclusive, and cycles outside both ranges are excluded.

    For every qualifying initialization cycle, consecutive changes are
    calculated at ``hh``-hour intervals from lead 0 through ``lead_time``
    inclusive. For example, ``hh=6`` and ``lead_time=18`` produce changes
    over 0--6, 6--12, and 12--18 hours. The Pearson correlation between the
    model changes and the corresponding best-track changes is calculated
    separately for each model and cycle.

    The returned mean and population standard deviation are calculated
    across all finite cycle correlations from all selected storms. Thus,
    each initialization cycle is one statistical case. Correlation is
    undefined for a cycle when fewer than two valid change pairs are
    available or either change vector is constant; such a cycle is not
    counted for that model.

    In strict mode, a cycle must have the requested metric at every lead
    from 0 through ``lead_time`` in the B-deck and for every requested model.
    In relaxed mode, all models still use the same available change
    intervals, but incomplete intervals are omitted independently for each
    cycle.

    Entries in ``storms`` may be explicit A-deck filenames such as
    ``"aep172026.dat"`` or basin/year patterns such as ``"aepALL2026.dat"``.

    Parameters
    ----------
    storms : list[str]
        A-deck filenames and/or ``ALL`` basin/year patterns.
    models : list[str]
        Non-empty list of four-character ATCF model identifiers.
    metric : str
        Forecast/best-track metric whose correlation is calculated.
    threshold_value_1 : float
        Exclusive lower VMAX threshold for the left group.
    threshold_value_2 : float
        Exclusive upper VMAX threshold for the left group.
    threshold_value_3 : float
        Exclusive lower VMAX threshold for the right group.
    threshold_value_4 : float
        Exclusive upper VMAX threshold for the right group. Each lower bound
        must be less than its corresponding upper bound. If the two ranges
        overlap, qualifying cycles contribute to both groups.
    hh : int
        Interval in hours over which consecutive changes are calculated.
    lead_time : int
        Inclusive endpoint of the correlation window. It must be an exact
        multiple of ``hh`` and at least ``2 * hh``. Leads are 0, ``hh``,
        ..., ``lead_time``.
    sample_mode : {"strict", "relaxed"}, default="strict"
        Homogeneous sampling rule.

    Returns
    -------
    mean_corr : list[list[float]]
        Two-by-M array-like list. Row 0 is the first VMAX range and row 1 is
        the second VMAX range; columns follow ``models``.
    corr_std : list[list[float]]
        Population standard deviation of the per-cycle correlations, with
        the same two-by-M layout.
    n_cycles : list[list[int]]
        Number of finite per-cycle correlations contributing to each value.
    """
    if not isinstance(storms, list) or not storms:
        raise ValueError("storms must be a non-empty list of A-deck filenames.")
    normalized_storms = [str(storm).strip().lower() for storm in storms]
    if len(set(normalized_storms)) != len(normalized_storms):
        raise ValueError("storms must not contain duplicates.")

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

    metric = str(metric).strip().upper()
    sample_mode = str(sample_mode).strip().lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {sorted(SUPPORTED_METRICS)}.")
    threshold_values = []
    for name, value in (
        ("threshold_value_1", threshold_value_1),
        ("threshold_value_2", threshold_value_2),
        ("threshold_value_3", threshold_value_3),
        ("threshold_value_4", threshold_value_4),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        threshold_values.append(number)
    (
        range_1_lower,
        range_1_upper,
        range_2_lower,
        range_2_upper,
    ) = threshold_values
    if range_1_lower >= range_1_upper or range_2_lower >= range_2_upper:
        raise ValueError(
            "thresholds must satisfy threshold_value_1 < threshold_value_2 "
            "and threshold_value_3 < threshold_value_4."
        )
    if isinstance(hh, bool) or not isinstance(hh, int) or hh <= 0:
        raise ValueError("hh must be a positive integer.")
    if (
        isinstance(lead_time, bool)
        or not isinstance(lead_time, int)
        or lead_time < 2 * hh
    ):
        raise ValueError(
            "lead_time must be an integer greater than or equal to 2 * hh "
            "so that each complete cycle has at least two change pairs."
        )
    if lead_time % hh:
        raise ValueError("lead_time must be an exact multiple of hh.")
    if sample_mode not in {"strict", "relaxed"}:
        raise ValueError("sample_mode must be either 'strict' or 'relaxed'.")

    adeck_paths = []
    seen_paths = set()
    storm_pattern = re.compile(r"a(al|wp|ep|cp)(\d{2}|all)(\d{4})\.dat")
    explicit_pattern = re.compile(r"a(al|wp|ep|cp)\d{2}\d{4}\.dat")
    for storm in normalized_storms:
        match = storm_pattern.fullmatch(storm)
        if match is None:
            raise ValueError(
                "Every storm must look like 'aep172026.dat' or "
                "'aepALL2026.dat'."
            )
        basin_code, storm_number, year = match.groups()
        if storm_number == "all":
            matching_paths = sorted(
                path
                for path in ADECK_DIR.glob(f"a{basin_code}??{year}.dat")
                if path.is_file() and explicit_pattern.fullmatch(path.name)
            )
            if not matching_paths:
                raise FileNotFoundError(
                    "No A-deck files found for "
                    f"basin={basin_code!r}, year={year!r}."
                )
        else:
            path = ADECK_DIR / storm
            if not path.is_file():
                raise FileNotFoundError(f"A-deck file not found: {path}")
            matching_paths = [path]

        for path in matching_paths:
            if path not in seen_paths:
                adeck_paths.append(path)
                seen_paths.add(path)

    n_models = len(normalized_models)
    lead_hours = np.arange(0, lead_time + hh, hh, dtype=int)
    lead_intervals = list(zip(lead_hours[:-1], lead_hours[1:]))
    correlations = [[[] for _ in range(n_models)] for _ in range(2)]

    for adeck_path in adeck_paths:
        bdeck_path = BDECK_DIR / f"b{adeck_path.name[1:]}"
        if not bdeck_path.is_file():
            raise FileNotFoundError(f"B-deck file not found: {bdeck_path}")

        forecast = abdeck._read_adeck_metric(
            adeck_path, metric, set(normalized_models)
        )
        best_track = abdeck._read_bdeck_metric(bdeck_path, metric)
        initial_vmax = abdeck._read_bdeck_metric(bdeck_path, "VMAX")
        bdeck_times = abdeck._read_bdeck_times(bdeck_path)

        cycles_by_model = [
            {
                cycle
                for cycle, record_model, _ in forecast
                if record_model == model
            }
            for model in normalized_models
        ]
        candidate_cycles = set.intersection(*cycles_by_model)

        def has_complete_point(cycle, tau: int) -> bool:
            return (
                cycle + timedelta(hours=tau) in best_track
                and all(
                    (cycle, model, tau) in forecast
                    for model in normalized_models
                )
            )

        cycle_groups = [
            sorted(
                cycle
                for cycle in candidate_cycles
                if cycle in initial_vmax
                and range_1_lower < initial_vmax[cycle] < range_1_upper
            ),
            sorted(
                cycle
                for cycle in candidate_cycles
                if cycle in initial_vmax
                and range_2_lower < initial_vmax[cycle] < range_2_upper
            ),
        ]

        for group_index, group_cycles in enumerate(cycle_groups):
            if sample_mode == "strict":
                cycles_and_intervals = [
                    (cycle, lead_intervals)
                    for cycle in group_cycles
                    if cycle + timedelta(hours=lead_time) in bdeck_times
                    and all(
                        has_complete_point(cycle, int(tau))
                        for tau in lead_hours
                    )
                ]
            else:
                cycles_and_intervals = []
                for cycle in group_cycles:
                    valid_intervals = [
                        (start_tau, end_tau)
                        for start_tau, end_tau in lead_intervals
                        if has_complete_point(cycle, int(start_tau))
                        and has_complete_point(cycle, int(end_tau))
                    ]
                    if valid_intervals:
                        cycles_and_intervals.append((cycle, valid_intervals))

            for model_index, model in enumerate(normalized_models):
                for cycle, valid_intervals in cycles_and_intervals:
                    forecast_changes = []
                    observed_changes = []
                    for start_tau, end_tau in valid_intervals:
                        start_tau = int(start_tau)
                        end_tau = int(end_tau)
                        forecast_changes.append(
                            forecast[(cycle, model, end_tau)]
                            - forecast[(cycle, model, start_tau)]
                        )
                        observed_changes.append(
                            best_track[
                                cycle + timedelta(hours=end_tau)
                            ]
                            - best_track[
                                cycle + timedelta(hours=start_tau)
                            ]
                        )
                    cycle_corr = abdeck._pearson_correlation(
                        np.asarray(forecast_changes, dtype=float),
                        np.asarray(observed_changes, dtype=float),
                    )
                    if np.isfinite(cycle_corr):
                        correlations[group_index][model_index].append(
                            cycle_corr
                        )

    mean_corr = np.full((2, n_models), np.nan, dtype=float)
    corr_std = np.full((2, n_models), np.nan, dtype=float)
    n_cycles = np.zeros((2, n_models), dtype=int)
    for group_index in range(2):
        for model_index in range(n_models):
            values = np.asarray(
                correlations[group_index][model_index], dtype=float
            )
            n_cycles[group_index, model_index] = values.size
            if values.size:
                mean_corr[group_index, model_index] = np.mean(values)
                corr_std[group_index, model_index] = np.std(values, ddof=0)

    return mean_corr.tolist(), corr_std.tolist(), n_cycles.tolist()
