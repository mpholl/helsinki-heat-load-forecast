"""Walk-forward evaluation of the day-ahead temperature forecast.

Every day of the evaluation year with 30 days of history is forecast from the
data before it and scored against the observations (MAE, RMSE, bias and
coverage of the 5th-95th percentile band). Also holds evaluation helpers shared
with the other evaluation scripts.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_heat_load import (
    DATA_FILE,
    ENSEMBLE_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    fit_seasonal_baseline,
    load_baseline_temperatures,
    load_temperatures,
    predict_temperature_ensemble,
)


def ensemble_crps(ensemble: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """CRPS per column: E|X - y| - 0.5 E|X - X'|, using the sorted-ensemble identity."""
    members = np.sort(ensemble, axis=0)
    size = members.shape[0]
    ranks = np.arange(1, size + 1)[:, np.newaxis]
    spread = np.sum((2 * ranks - size - 1) * members, axis=0) / size**2
    return np.mean(np.abs(ensemble - actual), axis=0) - spread


def forecast_skip_reason(
    temperatures: pd.Series, forecast_date: pd.Timestamp
) -> str | None:
    forecast_start = forecast_date.normalize()
    history_start = forecast_start - pd.Timedelta(days=30)
    previous_hour = forecast_start - pd.Timedelta(hours=1)
    if temperatures.index.min() > history_start:
        return (
            f"Incomplete 30-day lookback: data starts at "
            f"{temperatures.index.min()}."
        )
    if previous_hour not in temperatures.index:
        return f"Missing starting temperature at {previous_hour}."

    changes = temperatures.diff()
    consecutive_hours = temperatures.index.to_series().diff().eq(
        pd.Timedelta(hours=1)
    )
    in_history = (
        (temperatures.index >= history_start)
        & (temperatures.index < forecast_start)
    )
    hourly_changes = changes.loc[in_history & consecutive_hours.to_numpy()]
    counts = hourly_changes.groupby(hourly_changes.index.hour).count().reindex(
        range(24), fill_value=0
    )
    if (counts < 2).any():
        missing_hours = counts.index[counts < 2].tolist()
        return (
            "Fewer than two historical hourly changes for hours "
            f"{missing_hours} in the 30-day lookback."
        )
    return None


def evaluate_year(
    data_file: Path = DATA_FILE,
    output_dir: Path | None = None,
    ensemble_size: int = ENSEMBLE_SIZE,
    seed: int = RANDOM_SEED,
    seasonal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temperatures = load_temperatures(data_file)
    # The baseline is fitted on 2023-2024 only, so the evaluated year is out of sample.
    seasonal_coefficients = (
        fit_seasonal_baseline(load_baseline_temperatures()) if seasonal else None
    )
    if temperatures.empty:
        raise ValueError(f"No temperature observations found in {data_file}.")

    year = int(temperatures.index[0].year)
    output_dir = output_dir or RESULTS_DIR
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    hourly_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []

    for forecast_date in dates:
        reason = forecast_skip_reason(temperatures, forecast_date)
        if reason is not None:
            skipped_rows.append(
                {"date": forecast_date.strftime("%Y-%m-%d"), "reason": reason}
            )
            continue

        forecast_times, ensemble = predict_temperature_ensemble(
            temperatures,
            forecast_date.strftime("%Y-%m-%d"),
            ensemble_size=ensemble_size,
            seed=seed + forecast_date.dayofyear,
            seasonal_coefficients=seasonal_coefficients,
        )
        actual = temperatures.reindex(forecast_times)
        valid = actual.notna().to_numpy()
        if not valid.any():
            skipped_rows.append(
                {
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "reason": "No actual hourly temperatures available for this date.",
                }
            )
            continue

        observed_ensemble = ensemble[:, valid]
        observed_actual = actual.to_numpy(dtype=float)[valid]
        mean = observed_ensemble.mean(axis=0)
        median = np.median(observed_ensemble, axis=0)
        lower, upper = np.percentile(observed_ensemble, [5, 95], axis=0)
        errors = mean - observed_actual
        hourly = pd.DataFrame(
            {
                "date": forecast_date.strftime("%Y-%m-%d"),
                "actual_temperature_c": observed_actual,
                "ensemble_mean_c": mean,
                "ensemble_median_c": median,
                "ensemble_5th_percentile_c": lower,
                "ensemble_95th_percentile_c": upper,
                "mean_forecast_error_c": errors,
            },
            index=forecast_times[valid],
        )
        hourly.index.name = "timestamp_utc"
        hourly_rows.append(hourly)

        daily_ensemble_mean = observed_ensemble.mean(axis=1)
        daily_mean = float(daily_ensemble_mean.mean())
        daily_lower, daily_upper = np.percentile(daily_ensemble_mean, [5, 95])
        daily_actual_mean = float(np.mean(observed_actual))
        daily_rows.append(
            {
                "date": forecast_date.strftime("%Y-%m-%d"),
                "hours_evaluated": int(valid.sum()),
                "actual_daily_mean_c": daily_actual_mean,
                "forecast_daily_mean_c": daily_mean,
                "forecast_daily_5th_percentile_c": float(daily_lower),
                "forecast_daily_95th_percentile_c": float(daily_upper),
                "mae_c": float(np.mean(np.abs(errors))),
                "rmse_c": float(np.sqrt(np.mean(errors**2))),
                "mean_error_c": float(np.mean(errors)),
                "actual_hours_within_5th_95th_percentile": float(
                    np.mean((observed_actual >= lower) & (observed_actual <= upper))
                ),
            }
        )

    if not hourly_rows:
        raise ValueError(
            "No dates could be evaluated. Ensure the data includes at least "
            "30 days of history before forecast dates."
        )

    hourly_results = pd.concat(hourly_rows).sort_index()
    daily_results = pd.DataFrame(daily_rows).set_index("date")
    skipped_results = pd.DataFrame(skipped_rows, columns=["date", "reason"])

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_file = output_dir / f"temperature_yearly_hourly_{year}.csv"
    daily_file = output_dir / f"temperature_yearly_daily_{year}.csv"
    skipped_file = output_dir / f"temperature_yearly_skipped_{year}.csv"
    plot_file = output_dir / f"temperature_yearly_evaluation_{year}.png"
    hourly_results.to_csv(hourly_file, float_format="%.4f")
    daily_results.to_csv(daily_file, float_format="%.4f")
    skipped_results.to_csv(skipped_file, index=False)

    errors = hourly_results["mean_forecast_error_c"].to_numpy()
    actual = hourly_results["actual_temperature_c"].to_numpy()
    overall_coverage = np.mean(
        (
            actual
            >= hourly_results["ensemble_5th_percentile_c"].to_numpy()
        )
        & (
            actual
            <= hourly_results["ensemble_95th_percentile_c"].to_numpy()
        )
    )
    print(f"Walk-forward evaluation for {year}")
    print(
        f"Evaluated {len(daily_results)} of {len(dates)} days "
        f"({len(hourly_results)} observed hourly values); "
        f"skipped {len(skipped_results)} days."
    )
    print(f"Hourly ensemble-mean MAE: {np.mean(np.abs(errors)):.2f} °C")
    print(f"Hourly ensemble-mean RMSE: {np.sqrt(np.mean(errors**2)):.2f} °C")
    print(f"Hourly ensemble-mean bias: {np.mean(errors):.2f} °C")
    print(f"Hourly 5th-95th percentile coverage: {overall_coverage:.1%}")
    if not skipped_results.empty:
        print("\nSkipped dates:")
        print(skipped_results.to_string(index=False))

    plot_dates = pd.to_datetime(daily_results.index)
    plt.figure(figsize=(12, 6))
    plt.fill_between(
        plot_dates,
        daily_results["forecast_daily_5th_percentile_c"].to_numpy(),
        daily_results["forecast_daily_95th_percentile_c"].to_numpy(),
        alpha=0.25,
        label="Forecast 5th-95th percentile (daily mean)",
    )
    plt.plot(
        plot_dates,
        daily_results["forecast_daily_mean_c"],
        label="Ensemble forecast daily mean",
    )
    plt.plot(
        plot_dates,
        daily_results["actual_daily_mean_c"],
        label="Actual daily mean",
    )
    plt.xlabel("Date")
    plt.ylabel("Daily mean temperature [°C]")
    plt.title(f"Walk-forward temperature forecasts vs actuals ({year})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"\nSaved hourly results to {hourly_file}")
    print(f"Saved daily metrics to {daily_file}")
    print(f"Saved skipped dates to {skipped_file}")
    print(f"Saved annual comparison plot to {plot_file}")
    return hourly_results, daily_results, skipped_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 30-day-lookback ensemble forecast for each eligible "
            "date in the weather data and evaluate it against observations."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="CSV file containing hourly temperature observations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation CSVs and plot (default: results/).",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=ENSEMBLE_SIZE,
        help=f"Simulations per forecast day (default: {ENSEMBLE_SIZE}).",
    )
    parser.add_argument(
        "--no-seasonal-baseline",
        action="store_true",
        help="Disable the seasonal forcing in the Langevin model.",
    )
    args = parser.parse_args()
    evaluate_year(
        args.data,
        args.output_dir,
        args.ensemble_size,
        seasonal=not args.no_seasonal_baseline,
    )


if __name__ == "__main__":
    main()
