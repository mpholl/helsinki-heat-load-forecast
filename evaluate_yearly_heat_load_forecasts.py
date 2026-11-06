"""Walk-forward evaluation of the heat load forecast over the weather data year.

For every eligible day the Langevin temperature ensemble is converted to a heat
load ensemble exactly as in predict_heat_load.py. Because the heat load is
synthetic, the reference ("true") load is generated from the observed
temperature the same way as the synthetic data (simulate_true_load in
heat_load_synth.py).

A "same as yesterday" baseline (the true load 24 hours earlier) is scored
alongside for context.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_yearly_temperature_forecasts import ensemble_crps, forecast_skip_reason
from heat_load_synth import simulate_true_load
from predict_heat_load import (
    BLOCK_HOURS,
    DATA_FILE,
    ENSEMBLE_SIZE,
    HEAT_LOAD_DATA_FILE,
    NOISE,
    RANDOM_SEED,
    RESULTS_DIR,
    TEMPERATURE_WINDOW_C,
    fit_seasonal_baseline,
    load_baseline_temperatures,
    load_synthetic_heat_load,
    load_temperatures,
    predict_heat_load_ensemble,
    predict_temperature_ensemble,
)


COLD_DAY_THRESHOLD_C = 0.0


def hourly_summary(hourly: pd.DataFrame) -> pd.Series:
    errors = hourly["load_error"]
    true_load = hourly["true_load"]
    baseline_errors = hourly["yesterday_load"] - true_load
    return pd.Series(
        {
            "hours": len(hourly),
            "load_mae": errors.abs().mean(),
            "load_rmse": np.sqrt((errors**2).mean()),
            "load_bias": errors.mean(),
            "load_crps": hourly["load_crps"].mean(),
            "coverage_50": (
                (true_load >= hourly["load_p25"]) & (true_load <= hourly["load_p75"])
            ).mean(),
            "coverage_90": (
                (true_load >= hourly["load_p05"]) & (true_load <= hourly["load_p95"])
            ).mean(),
            "width_90": (hourly["load_p95"] - hourly["load_p05"]).mean(),
            "yesterday_mae": baseline_errors.abs().mean(),
        }
    )


def daily_summary(daily: pd.DataFrame) -> pd.Series:
    complete = daily.loc[daily["complete_day"]]
    errors = complete["forecast_energy"] - complete["true_energy"]
    baseline = complete["yesterday_energy"] - complete["true_energy"]
    return pd.Series(
        {
            "days": len(complete),
            "energy_mae": errors.abs().mean(),
            "energy_mape": (errors.abs() / complete["true_energy"]).mean(),
            "energy_bias": errors.mean(),
            "energy_coverage_90": (
                (complete["true_energy"] >= complete["energy_p05"])
                & (complete["true_energy"] <= complete["energy_p95"])
            ).mean(),
            "yesterday_energy_mae": baseline.abs().mean(),
            "yesterday_energy_mape": (baseline.abs() / complete["true_energy"]).mean(),
        }
    )


def evaluate_year(
    data_file: Path = DATA_FILE,
    heat_load_file: Path = HEAT_LOAD_DATA_FILE,
    output_dir: Path | None = None,
    ensemble_size: int = ENSEMBLE_SIZE,
    noise: str = NOISE,
    block_hours: int = BLOCK_HOURS,
    seasonal: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temperatures = load_temperatures(data_file)
    synthetic_heat_load = load_synthetic_heat_load(heat_load_file)
    observed_load = simulate_true_load(temperatures)
    seasonal_coefficients = (
        fit_seasonal_baseline(load_baseline_temperatures()) if seasonal else None
    )
    year = int(temperatures.index[0].year)
    output_dir = output_dir or RESULTS_DIR
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    hourly_rows: list[pd.DataFrame] = []
    daily_rows: list[dict[str, object]] = []
    skipped = 0
    for forecast_date in dates:
        if forecast_skip_reason(temperatures, forecast_date) is not None:
            skipped += 1
            continue
        date_label = forecast_date.strftime("%Y-%m-%d")
        forecast_times, temperature_ensemble = predict_temperature_ensemble(
            temperatures,
            date_label,
            ensemble_size=ensemble_size,
            seed=seed + forecast_date.dayofyear,
            seasonal_coefficients=seasonal_coefficients,
            noise=noise,
            block_hours=block_hours,
        )
        load_ensemble = predict_heat_load_ensemble(
            temperature_ensemble,
            synthetic_heat_load,
            temperature_window=TEMPERATURE_WINDOW_C,
            seed=seed + 1000 + forecast_date.dayofyear,
        )

        observed = temperatures.reindex(forecast_times).to_numpy(dtype=float)
        yesterday = temperatures.reindex(forecast_times - pd.Timedelta(days=1))
        valid = ~np.isnan(observed)
        if not valid.any():
            skipped += 1
            continue
        true_load = observed_load.reindex(forecast_times).to_numpy(dtype=float)
        yesterday_load = observed_load.reindex(
            forecast_times - pd.Timedelta(days=1)
        ).to_numpy(dtype=float)

        ensemble = load_ensemble[:, valid]
        p05, p25, p75, p95 = np.percentile(ensemble, [5, 25, 75, 95], axis=0)
        mean = ensemble.mean(axis=0)
        hourly_rows.append(
            pd.DataFrame(
                {
                    "date": date_label,
                    "lead_hours": np.arange(1, 25)[valid],
                    "observed_temperature_c": observed[valid],
                    "forecast_temperature_c": temperature_ensemble[:, valid].mean(axis=0),
                    "true_load": true_load[valid],
                    "forecast_load": mean,
                    "load_p05": p05,
                    "load_p25": p25,
                    "load_p75": p75,
                    "load_p95": p95,
                    "load_error": mean - true_load[valid],
                    "load_crps": ensemble_crps(ensemble, true_load[valid]),
                    "yesterday_load": yesterday_load[valid],
                },
                index=pd.Index(forecast_times[valid], name="timestamp_utc"),
            )
        )

        # Daily energy is the sum of the 24 hourly loads [load x h]; only
        # complete days (24 observed hours, also yesterday) are scored.
        complete = bool(valid.all() and yesterday.notna().all())
        energy = load_ensemble.sum(axis=1)
        daily_rows.append(
            {
                "date": date_label,
                "complete_day": complete,
                "observed_mean_temperature_c": float(np.nanmean(observed)),
                "true_energy": float(true_load.sum()) if complete else np.nan,
                "forecast_energy": float(energy.mean()),
                "energy_p05": float(np.percentile(energy, 5)),
                "energy_p95": float(np.percentile(energy, 95)),
                "yesterday_energy": float(yesterday_load.sum()) if complete else np.nan,
            }
        )

    hourly = pd.concat(hourly_rows)
    daily = pd.DataFrame(daily_rows)
    summary = pd.concat([hourly_summary(hourly), daily_summary(daily)])

    hourly["month"] = hourly.index.month
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    by_month = pd.concat(
        [
            hourly.groupby("month").apply(hourly_summary, include_groups=False),
            daily.groupby("month").apply(daily_summary, include_groups=False),
        ],
        axis=1,
    )
    cold_dates = daily.loc[
        daily["observed_mean_temperature_c"] < COLD_DAY_THRESHOLD_C, "date"
    ]
    cold = pd.concat(
        [
            hourly_summary(hourly.loc[hourly["date"].isin(cold_dates)]),
            daily_summary(daily.loc[daily["date"].isin(cold_dates)]),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_file = output_dir / f"heat_load_yearly_hourly_{year}.csv"
    daily_file = output_dir / f"heat_load_yearly_daily_{year}.csv"
    summary_file = output_dir / f"heat_load_yearly_summary_{year}.csv"
    plot_file = output_dir / f"heat_load_yearly_evaluation_{year}.png"
    hourly.drop(columns="month").to_csv(hourly_file, float_format="%.4f")
    daily.drop(columns="month").to_csv(daily_file, index=False, float_format="%.4f")
    pd.concat({"all days": summary, "cold days": cold}, axis=1).to_csv(
        summary_file, float_format="%.4f"
    )

    plot_daily = daily.loc[daily["complete_day"]]
    plot_dates = pd.to_datetime(plot_daily["date"])
    figure, (energy_axis, error_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1]
    )
    energy_axis.fill_between(
        plot_dates,
        plot_daily["energy_p05"],
        plot_daily["energy_p95"],
        alpha=0.25,
        label="Forecast 5th-95th percentile",
    )
    energy_axis.plot(plot_dates, plot_daily["forecast_energy"], label="Forecast mean")
    energy_axis.plot(plot_dates, plot_daily["true_energy"], color="black", lw=1, label="True")
    energy_axis.set(ylabel="Daily heat energy [load x h]", title=f"Daily heat energy, walk-forward {year}")
    energy_axis.legend()
    error_axis.axhline(0, color="black", lw=0.8)
    error_axis.plot(
        plot_dates,
        plot_daily["forecast_energy"] - plot_daily["true_energy"],
        label="Forecast",
    )
    error_axis.plot(
        plot_dates,
        plot_daily["yesterday_energy"] - plot_daily["true_energy"],
        alpha=0.6,
        label="Same as yesterday",
    )
    error_axis.set(xlabel="Date", ylabel="Error [load x h]")
    error_axis.legend()
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)

    print(f"Walk-forward heat load evaluation for {year}")
    print(
        f"Model: Langevin, {noise} noise"
        + (f" ({block_hours} h blocks)" if noise == "block" else "")
        + (", seasonal forcing" if seasonal else ", no seasonal forcing")
    )
    print(f"Evaluated {daily.shape[0]} days ({len(hourly)} hours), skipped {skipped}.\n")
    print(pd.concat({"all days": summary, "cold days": cold}, axis=1).round(3).to_string())
    print(f"\n(cold days: observed daily mean below {COLD_DAY_THRESHOLD_C:.0f} °C)")
    print("\nBy month:")
    print(
        by_month[
            ["load_mae", "load_crps", "coverage_90", "energy_mape", "energy_coverage_90", "yesterday_energy_mape"]
        ].round(3).to_string()
    )
    print(f"\nSaved hourly results to {hourly_file}")
    print(f"Saved daily results to {daily_file}")
    print(f"Saved summary to {summary_file}")
    print(f"Saved plot to {plot_file}")
    return hourly, daily, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the synthetic heat load forecast."
    )
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--heat-load-data", type=Path, default=HEAT_LOAD_DATA_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/).",
    )
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--noise", choices=("block", "gaussian"), default=NOISE)
    parser.add_argument("--block-hours", type=int, default=BLOCK_HOURS)
    parser.add_argument("--no-seasonal-baseline", action="store_true")
    args = parser.parse_args()
    evaluate_year(
        args.data,
        args.heat_load_data,
        args.output_dir,
        args.ensemble_size,
        noise=args.noise,
        block_hours=args.block_hours,
        seasonal=not args.no_seasonal_baseline,
    )


if __name__ == "__main__":
    main()
