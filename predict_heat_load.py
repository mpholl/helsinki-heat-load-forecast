"""Day-ahead temperature and heat load forecast for Helsinki Kaisaniemi.

The temperature for the next 24 hours is simulated as an ensemble with a
Langevin (Ornstein-Uhlenbeck) model fitted to the previous 30 days, with
seasonal forcing from a climatology fitted on 2023-2024 and noise resampled
from blocks of the model's own past errors. Each simulated temperature path is
then converted to a heat load path using the synthetic temperature-load data
from heat_load_synth.py.

Usage: python predict_heat_load.py --date 2025-12-23
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heat_load_synth import simulate_true_load


DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def temperature_data_file(year: int) -> Path:
    """Hourly FMI observations (UTC) for one year."""
    return DATA_DIR / f"helsinki_kaisaniemi_{year}.csv"


# Year that is forecast and evaluated.
DATA_FILE = temperature_data_file(2025)
# Earlier years used only to fit the seasonal baseline, so the 2025 forecasts
# and their evaluation never see 2025 data through the baseline.
BASELINE_YEARS = (2023, 2024)
HEAT_LOAD_DATA_FILE = DATA_DIR / "synthetic_heat_load.csv"
FORECAST_DATE = "2025-12-23"
ENSEMBLE_SIZE = 2000
RANDOM_SEED = 0
TEMPERATURE_WINDOW_C = 0.5
# Half-life in days of the recency weights used for the diurnal profile.
# Walk-forward tests on 2025 found 3-7 days roughly equally good (best ~4-5).
PROFILE_HALF_LIFE_DAYS = 5.0
# Noise model of the Langevin simulation: "gaussian" (independent draws) or
# "block" (blocks of consecutive past errors, BLOCK_HOURS long). Block noise
# with seasonal forcing gave 88.8% coverage of the 90% interval on 2025,
# against 78.6% with Gaussian noise (compare_langevin_variants.py).
NOISE = "block"
BLOCK_HOURS = 6


def load_temperatures(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    date_parts = data[["Year", "Month", "Day"]].rename(
        columns={"Year": "year", "Month": "month", "Day": "day"}
    )
    date_parts = date_parts.astype(int)
    date_parts["hour"] = data["Time [UTC]"].str[:2].astype(int)
    timestamps = pd.to_datetime(date_parts)
    temperatures = pd.to_numeric(data["Average temperature [°C]"], errors="raise")
    series = pd.Series(temperatures.to_numpy(), index=timestamps, name="temperature")
    series = series.sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Weather data in {path} contains duplicate timestamps.")
    return series


def seasonal_design_matrix(index: pd.DatetimeIndex) -> np.ndarray:
    """Regressors for the seasonal baseline (17 columns).

    Annual cycle with two harmonics, diurnal cycle with two harmonics, and the
    product of each diurnal term with the first annual harmonic so that the
    size and timing of the daily cycle can change through the year.
    """
    day_of_year = index.dayofyear.to_numpy() - 1 + index.hour.to_numpy() / 24
    annual_angle = 2 * np.pi * day_of_year / 365.25
    diurnal_angle = 2 * np.pi * index.hour.to_numpy() / 24
    annual = [
        np.cos(annual_angle),
        np.sin(annual_angle),
        np.cos(2 * annual_angle),
        np.sin(2 * annual_angle),
    ]
    diurnal = [
        np.cos(diurnal_angle),
        np.sin(diurnal_angle),
        np.cos(2 * diurnal_angle),
        np.sin(2 * diurnal_angle),
    ]
    interactions = [d * a for d in diurnal for a in annual[:2]]
    return np.column_stack([np.ones(len(index)), *annual, *diurnal, *interactions])


def fit_seasonal_baseline(temperatures: pd.Series) -> np.ndarray:
    """Least-squares fit of the seasonal baseline; returns its coefficients."""
    valid = temperatures.notna().to_numpy()
    design = seasonal_design_matrix(temperatures.index[valid])
    coefficients, *_ = np.linalg.lstsq(
        design, temperatures.to_numpy(dtype=float)[valid], rcond=None
    )
    return coefficients


def load_baseline_temperatures(years: tuple[int, ...] = BASELINE_YEARS) -> pd.Series:
    """Concatenate the years used to fit the seasonal baseline."""
    paths = [temperature_data_file(year) for year in years]
    series = pd.concat([load_temperatures(path) for path in paths]).sort_index()
    if series.index.has_duplicates:
        raise ValueError("Baseline data files overlap in time.")
    return series


def seasonal_baseline(
    index: pd.DatetimeIndex, coefficients: np.ndarray | None
) -> np.ndarray:
    """Evaluate the seasonal baseline at the given times (zero if no coefficients)."""
    if coefficients is None:
        return np.zeros(len(index))
    return seasonal_design_matrix(index) @ coefficients


def sample_error_blocks(
    innovations: pd.Series,
    history_start: pd.Timestamp,
    forecast_start: pd.Timestamp,
    block_hours: int,
    ensemble_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build (ensemble_size, 24) noise paths from blocks of past errors.

    Blocks are aligned to the hour of day, so forecast hours 06-11 copy the
    errors of hours 06-11 of some past day. Days with a gap inside a block are
    not used for that block.
    """
    window = pd.date_range(history_start, forecast_start, freq="h", inclusive="left")
    errors_by_day = innovations.reindex(window).to_numpy().reshape(-1, 24)
    paths = np.empty((ensemble_size, 24))
    for start in range(0, 24, block_hours):
        block = slice(start, start + block_hours)
        complete_days = np.flatnonzero(~np.isnan(errors_by_day[:, block]).any(axis=1))
        if len(complete_days) < 2:
            raise ValueError(
                f"Fewer than two complete past days of errors for hours "
                f"{start:02d}-{start + block_hours - 1:02d}."
            )
        paths[:, block] = errors_by_day[rng.choice(complete_days, ensemble_size), block]
    return paths


def predict_temperature_ensemble(
    temperatures: pd.Series,
    forecast_date: str,
    ensemble_size: int = ENSEMBLE_SIZE,
    seed: int = RANDOM_SEED,
    profile_half_life: float | None = PROFILE_HALF_LIFE_DAYS,
    seasonal_coefficients: np.ndarray | None = None,
    noise: str = NOISE,
    block_hours: int = BLOCK_HOURS,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Simulate the next day with a discretised Langevin (Ornstein-Uhlenbeck) model.

    Seasonal forcing: if seasonal_coefficients (from fit_seasonal_baseline) are
    given, the seasonal baseline B(t) is first subtracted from the temperatures
    and added back to the simulated values, so the process relaxes towards a
    target that moves with the season. Without them B(t) = 0.

    The deseasonalised temperature x = T - B(t) is split into a diurnal
    profile mu(h), the mean of each hour of day over the previous 30 days, and
    an anomaly a = x - mu(h) that relaxes back towards zero. The profile is a
    recency-weighted mean with weights 0.5 ** (age_days / profile_half_life);
    pass None for an unweighted mean. With the baseline, mu(h) captures the
    recent offset from the seasonal norm. phi and the noise are fitted
    unweighted over the full window (weighting them made walk-forward scores worse).

    The temperature therefore follows T(t) = B(t) + mu(h) + a(t), and the
    anomaly follows:

        da = -theta * a * dt + sigma(h) * dW

    With an hourly step the exact discretisation is the AR(1) recursion
    a_k = phi * a_{k-1} + eps_k, where phi = exp(-theta) and
    eps_k ~ N(0, sigma_eps(h)^2). phi is estimated by least squares on
    consecutive anomaly pairs; the noise variance is estimated per hour of day.

    Noise: with noise="gaussian" each eps_k is drawn independently from
    N(0, sigma_eps(h)^2). With noise="block" the eps_k are copied from the
    model's own past errors (eps = a_k - phi * a_{k-1}) in the 30-day window:
    the day is split into blocks of block_hours hours, and for each block every
    member copies that block's consecutive errors from a randomly chosen past
    day. This keeps the heavy tails, the hour-of-day pattern and, within a
    block, the serial correlation of real errors (runs of same-sign errors when
    a front passes), which independent Gaussian draws lose.
    """
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be greater than zero.")
    if noise not in ("gaussian", "block"):
        raise ValueError("noise must be 'gaussian' or 'block'.")
    if noise == "block" and (block_hours <= 0 or 24 % block_hours != 0):
        raise ValueError("block_hours must be a positive divisor of 24.")
    if profile_half_life is not None and profile_half_life <= 0:
        raise ValueError("profile_half_life must be greater than zero or None.")

    forecast_start = pd.Timestamp(forecast_date).normalize()
    history_start = forecast_start - pd.Timedelta(days=30)
    previous_hour = forecast_start - pd.Timedelta(hours=1)
    if previous_hour not in temperatures.index:
        raise ValueError(f"Missing starting temperature at {previous_hour}.")

    observed = temperatures
    temperatures = observed - seasonal_baseline(observed.index, seasonal_coefficients)

    in_history = (
        (temperatures.index >= history_start)
        & (temperatures.index < forecast_start)
    )
    history = temperatures.loc[in_history]
    if profile_half_life is None:
        weights = np.ones(len(history))
    else:
        age_days = (forecast_start - history.index).total_seconds() / 86400
        weights = 0.5 ** (age_days.to_numpy() / profile_half_life)
    weighted = pd.DataFrame(
        {"weighted_sum": history.to_numpy() * weights, "weight": weights},
        index=history.index,
    ).groupby(history.index.hour).sum()
    profile = (weighted["weighted_sum"] / weighted["weight"]).reindex(range(24))
    if profile.isna().any():
        raise ValueError(
            "The previous 30 days must contain observations for every hour of the day."
        )
    profile_values = profile.to_numpy(dtype=float)

    anomalies = temperatures - profile_values[temperatures.index.hour]
    consecutive_hours = temperatures.index.to_series().diff().eq(
        pd.Timedelta(hours=1)
    )
    pair_mask = in_history & consecutive_hours.to_numpy()
    current = anomalies.loc[pair_mask]
    previous = anomalies.shift(1).loc[pair_mask]
    denominator = float(np.dot(previous, previous))
    phi = float(np.dot(previous, current) / denominator) if denominator > 0 else 0.0
    # phi must stay in [0, 1) for a mean-reverting (theta > 0) process.
    phi = float(np.clip(phi, 0.0, 0.995))

    innovations = current - phi * previous
    statistics = innovations.groupby(innovations.index.hour).agg(["count", "var"])
    statistics = statistics.reindex(range(24))
    if statistics["count"].isna().any() or (statistics["count"] < 2).any():
        raise ValueError(
            "The previous 30 days must contain at least two hourly changes "
            "for every hour of the day."
        )
    noise_sd = np.sqrt(statistics["var"].to_numpy(dtype=float))

    forecast_times = pd.date_range(forecast_start, periods=24, freq="h")
    forecast_baseline = seasonal_baseline(forecast_times, seasonal_coefficients)
    rng = np.random.default_rng(seed)
    if noise == "gaussian":
        noise_paths = rng.normal(
            0.0, noise_sd[forecast_times.hour], (ensemble_size, len(forecast_times))
        )
    else:
        noise_paths = sample_error_blocks(
            innovations, history_start, forecast_start, block_hours, ensemble_size, rng
        )

    ensemble = np.empty((ensemble_size, len(forecast_times)))
    anomaly = np.full(
        ensemble_size,
        temperatures.loc[previous_hour] - profile_values[previous_hour.hour],
        dtype=float,
    )

    for step, timestamp in enumerate(forecast_times):
        anomaly = phi * anomaly + noise_paths[:, step]
        ensemble[:, step] = (
            forecast_baseline[step] + profile_values[timestamp.hour] + anomaly
        )

    return forecast_times, ensemble


def save_forecast(
    forecast_times: pd.DatetimeIndex,
    ensemble: np.ndarray,
    forecast_date: str,
    output_dir: Path,
) -> None:
    date_label = pd.Timestamp(forecast_date).strftime("%Y-%m-%d")
    ensemble_file = output_dir / f"temperature_ensemble_{date_label}.csv"
    plot_file = output_dir / f"temperature_forecast_{date_label}.png"

    ensemble_data = pd.DataFrame(
        ensemble.T,
        index=forecast_times,
        columns=[
            f"simulation_{number:04d}"
            for number in range(1, ensemble.shape[0] + 1)
        ],
    )
    ensemble_data.index.name = "timestamp_utc"
    ensemble_data.to_csv(ensemble_file)

    mean = ensemble.mean(axis=0)
    lower, upper = np.percentile(ensemble, [5, 95], axis=0)
    plt.figure(figsize=(10, 5))
    plt.fill_between(forecast_times, lower, upper, alpha=0.25, label="5th-95th percentile")
    plt.plot(forecast_times, mean, label="Ensemble mean")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Temperature forecast for {date_label} ({ensemble.shape[0]} simulations)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    summary = pd.DataFrame(
        {
            "mean": mean,
            "median": np.median(ensemble, axis=0),
            "5th_percentile": lower,
            "95th_percentile": upper,
        },
        index=forecast_times,
    )
    summary.index.name = "timestamp_utc"
    print(summary.round(2).to_string())
    print(f"\nSaved all simulations to {ensemble_file}")
    print(f"Saved forecast plot to {plot_file}")


def load_synthetic_heat_load(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {"temperature", "heat_load"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}."
        )

    heat_load_data = data[["temperature", "heat_load"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(heat_load_data.to_numpy()).all():
        raise ValueError(f"{path} contains non-finite temperature or heat load values.")
    if heat_load_data.empty:
        raise ValueError(f"No synthetic heat load data found in {path}.")
    return heat_load_data


def predict_heat_load_ensemble(
    temperature_ensemble: np.ndarray,
    heat_load_data: pd.DataFrame,
    temperature_window: float = TEMPERATURE_WINDOW_C,
    seed: int = RANDOM_SEED + 1,
) -> np.ndarray:
    """Draw one synthetic load within ±temperature_window of each member's temperature.

    Temperatures outside the synthetic data range are clipped to its edge.
    This is exact, not an approximation: the load curve is flat at both ends
    (minimum load above ~15 °C, maximum below ~-25 °C), so a member at 37 °C
    has the same load as one at 35 °C. Rare members of hot-day forecasts
    exceed the 35 °C limit.
    """
    if temperature_window <= 0:
        raise ValueError("temperature_window must be greater than zero.")
    if temperature_ensemble.ndim != 2:
        raise ValueError("temperature_ensemble must be a 2D array.")

    sorted_data = heat_load_data.sort_values("temperature")
    synthetic_temperatures = sorted_data["temperature"].to_numpy()
    synthetic_loads = sorted_data["heat_load"].to_numpy()
    minimum, maximum = synthetic_temperatures[[0, -1]]
    temperatures = np.clip(temperature_ensemble, minimum, maximum)

    start = np.searchsorted(
        synthetic_temperatures, temperatures - temperature_window, side="left"
    )
    stop = np.searchsorted(
        synthetic_temperatures, temperatures + temperature_window, side="right"
    )
    if (start == stop).any():
        temperature = temperatures[start == stop][0]
        raise ValueError(
            f"No synthetic heat load samples within ±{temperature_window:.2f} °C "
            f"of forecast temperature {temperature:.2f} °C."
        )
    rng = np.random.default_rng(seed)
    picks = start + np.floor(rng.random(temperatures.shape) * (stop - start)).astype(int)
    return synthetic_loads[picks]


def save_heat_load_forecast(
    forecast_times: pd.DatetimeIndex,
    heat_load_ensemble: np.ndarray,
    forecast_date: str,
    output_dir: Path,
    reference_load: np.ndarray | None = None,
) -> None:
    """Save and plot the heat load ensemble.

    reference_load, if given, is the synthetic load for the observed
    temperatures (simulate_true_load) and is drawn for comparison.
    """
    date_label = pd.Timestamp(forecast_date).strftime("%Y-%m-%d")
    ensemble_file = output_dir / f"heat_load_ensemble_{date_label}.csv"
    plot_file = output_dir / f"heat_load_forecast_{date_label}.png"

    ensemble_data = pd.DataFrame(
        heat_load_ensemble.T,
        index=forecast_times,
        columns=[
            f"simulation_{number:04d}"
            for number in range(1, heat_load_ensemble.shape[0] + 1)
        ],
    )
    ensemble_data.index.name = "timestamp_utc"
    ensemble_data.to_csv(ensemble_file)

    mean = heat_load_ensemble.mean(axis=0)
    lower, upper = np.percentile(heat_load_ensemble, [5, 95], axis=0)
    plt.figure(figsize=(10, 5))
    plt.fill_between(
        forecast_times,
        lower,
        upper,
        alpha=0.25,
        label="5th-95th percentile",
    )
    plt.plot(forecast_times, mean, label="Ensemble mean")
    if reference_load is not None:
        plt.plot(
            forecast_times,
            reference_load,
            marker="o",
            label="Reference load (from observed temperature)",
        )
    plt.xlabel("Time [UTC]")
    plt.ylabel("Synthetic heat load")
    plt.title(
        f"Heat load forecast for {date_label} "
        f"({heat_load_ensemble.shape[0]} simulations)"
    )
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    summary = pd.DataFrame(
        {
            "mean": mean,
            "median": np.median(heat_load_ensemble, axis=0),
            "5th_percentile": lower,
            "95th_percentile": upper,
        },
        index=forecast_times,
    )
    if reference_load is not None:
        summary["reference"] = reference_load
    summary.index.name = "timestamp_utc"
    print("\nSynthetic heat load forecast:")
    print(summary.round(3).to_string())
    print(f"\nSaved heat load simulations to {ensemble_file}")
    print(f"Saved heat load forecast plot to {plot_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast hourly temperature with the Langevin model and convert it "
            "to a synthetic heat load forecast."
        )
    )
    parser.add_argument(
        "--date",
        default=FORECAST_DATE,
        help=f"UTC date to forecast (default: {FORECAST_DATE})",
    )
    parser.add_argument(
        "--heat-load-data",
        type=Path,
        default=HEAT_LOAD_DATA_FILE,
        help="Synthetic temperature-to-heat-load data CSV.",
    )
    parser.add_argument(
        "--temperature-window",
        type=float,
        default=TEMPERATURE_WINDOW_C,
        help=(
            "Temperature radius in °C for sampling nearby synthetic loads "
            f"(default: ±{TEMPERATURE_WINDOW_C} °C)."
        ),
    )
    parser.add_argument(
        "--profile-half-life",
        type=float,
        default=PROFILE_HALF_LIFE_DAYS,
        help=(
            "Half-life in days of the recency weights for the diurnal profile; "
            f"0 disables weighting (default: {PROFILE_HALF_LIFE_DAYS})."
        ),
    )
    parser.add_argument(
        "--no-seasonal-baseline",
        action="store_true",
        help="Disable the seasonal forcing fitted from the 2023-2024 weather data.",
    )
    parser.add_argument(
        "--noise",
        choices=("block", "gaussian"),
        default=NOISE,
        help=f"Noise model of the Langevin simulation (default: {NOISE}).",
    )
    parser.add_argument(
        "--block-hours",
        type=int,
        default=BLOCK_HOURS,
        help=f"Block length for block noise, a divisor of 24 (default: {BLOCK_HOURS}).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    temperatures = load_temperatures(DATA_FILE)
    seasonal_coefficients = (
        None
        if args.no_seasonal_baseline
        else fit_seasonal_baseline(load_baseline_temperatures())
    )
    forecast_times, ensemble = predict_temperature_ensemble(
        temperatures,
        args.date,
        profile_half_life=args.profile_half_life or None,
        seasonal_coefficients=seasonal_coefficients,
        noise=args.noise,
        block_hours=args.block_hours,
    )
    save_forecast(forecast_times, ensemble, args.date, RESULTS_DIR)
    synthetic_heat_load = load_synthetic_heat_load(args.heat_load_data)
    heat_load_ensemble = predict_heat_load_ensemble(
        ensemble,
        synthetic_heat_load,
        temperature_window=args.temperature_window,
    )
    # Reference load from the observed temperatures, if the day has been observed.
    reference = simulate_true_load(temperatures).reindex(forecast_times)
    save_heat_load_forecast(
        forecast_times,
        heat_load_ensemble,
        args.date,
        RESULTS_DIR,
        reference_load=reference.to_numpy() if reference.notna().any() else None,
    )


if __name__ == "__main__":
    main()
