"""Compare one saved temperature forecast ensemble with the observations.

Reads results/temperature_ensemble_<date>.csv written by predict_heat_load.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_heat_load import DATA_FILE, FORECAST_DATE, RESULTS_DIR, load_temperatures


def compare_forecast(
    forecast_date: str,
    data_file: Path = DATA_FILE,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    date_label = pd.Timestamp(forecast_date).strftime("%Y-%m-%d")
    output_dir = output_dir or RESULTS_DIR
    ensemble_file = output_dir / f"temperature_ensemble_{date_label}.csv"
    if not ensemble_file.is_file():
        raise FileNotFoundError(
            f"Forecast ensemble not found: {ensemble_file}. "
            "Run predict_heat_load.py for this date first."
        )

    ensemble_data = pd.read_csv(ensemble_file, parse_dates=["timestamp_utc"])
    if "timestamp_utc" not in ensemble_data:
        raise ValueError(f"{ensemble_file} must contain a timestamp_utc column.")
    simulation_columns = [
        column for column in ensemble_data.columns if column.startswith("simulation_")
    ]
    if not simulation_columns:
        raise ValueError(f"No simulation columns found in {ensemble_file}.")

    forecast_times = pd.date_range(
        pd.Timestamp(forecast_date).normalize(), periods=24, freq="h"
    )
    ensemble_data = ensemble_data.set_index("timestamp_utc").sort_index()
    if ensemble_data.index.has_duplicates:
        raise ValueError(f"{ensemble_file} contains duplicate timestamps.")
    if not forecast_times.isin(ensemble_data.index).all():
        raise ValueError(
            f"{ensemble_file} must contain all 24 hourly timestamps for {date_label}."
        )
    ensemble = ensemble_data.loc[forecast_times, simulation_columns].to_numpy(
        dtype=float
    )

    actual_temperatures = load_temperatures(data_file)
    if not forecast_times.isin(actual_temperatures.index).all():
        raise ValueError(
            f"{data_file} must contain all 24 hourly observations for {date_label}."
        )
    actual = actual_temperatures.loc[forecast_times].to_numpy(dtype=float)

    mean = ensemble.mean(axis=1)
    lower, upper = np.percentile(ensemble, [5, 95], axis=1)
    errors = mean - actual
    comparison = pd.DataFrame(
        {
            "actual_temperature_c": actual,
            "ensemble_mean_c": mean,
            "ensemble_median_c": np.median(ensemble, axis=1),
            "ensemble_5th_percentile_c": lower,
            "ensemble_95th_percentile_c": upper,
            "mean_forecast_error_c": errors,
        },
        index=forecast_times,
    )
    comparison.index.name = "timestamp_utc"
    metrics = {
        "mae_c": float(np.mean(np.abs(errors))),
        "rmse_c": float(np.sqrt(np.mean(errors**2))),
        "mean_error_c": float(np.mean(errors)),
        "actual_within_5th_95th_percentile": float(
            np.mean((actual >= lower) & (actual <= upper))
        ),
    }

    comparison_file = output_dir / f"temperature_comparison_{date_label}.csv"
    plot_file = output_dir / f"temperature_comparison_{date_label}.png"
    comparison.to_csv(comparison_file, float_format="%.4f")

    plt.figure(figsize=(10, 5))
    plt.fill_between(
        forecast_times,
        lower,
        upper,
        alpha=0.25,
        label="Forecast 5th-95th percentile",
    )
    plt.plot(forecast_times, mean, label="Ensemble mean")
    plt.plot(forecast_times, actual, marker="o", label="Actual temperature")
    plt.xlabel("Time [UTC]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Temperature forecast vs actual for {date_label}")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"Forecast vs actual for {date_label} ({len(forecast_times)} hourly values)")
    for metric, value in metrics.items():
        if metric == "actual_within_5th_95th_percentile":
            print(f"{metric}: {value:.1%}")
        else:
            print(f"{metric}: {value:.2f} °C")
    print(f"\nSaved hourly comparison to {comparison_file}")
    print(f"Saved comparison plot to {plot_file}")
    return comparison, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a saved temperature ensemble forecast with observations."
    )
    parser.add_argument(
        "--date",
        default=FORECAST_DATE,
        help=f"UTC forecast date to compare (default: {FORECAST_DATE})",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_FILE,
        help="CSV file containing the actual hourly temperatures.",
    )
    args = parser.parse_args()
    compare_forecast(args.date, args.data)


if __name__ == "__main__":
    main()
