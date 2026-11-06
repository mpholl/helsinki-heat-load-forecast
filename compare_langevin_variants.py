"""Walk-forward comparison of the Langevin model variants.

Four variants: Gaussian or block-bootstrap noise, each with and without the
seasonal forcing (baseline fitted on 2023-2024). Every variant forecasts every
eligible day of the evaluation year from the 30 days before it, with the same
random seed per day, and is scored against the observations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_yearly_temperature_forecasts import ensemble_crps, forecast_skip_reason
from predict_heat_load import (
    BLOCK_HOURS,
    DATA_FILE,
    ENSEMBLE_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
    fit_seasonal_baseline,
    load_baseline_temperatures,
    load_temperatures,
    predict_temperature_ensemble,
)


VARIANTS = (
    ("Gaussian", False),
    ("Gaussian", True),
    ("Block", False),
    ("Block", True),
)
REFERENCE = "Gaussian, no seasonal"
LEAD_BINS = [0, 3, 6, 12, 18, 24]


def variant_name(noise: str, seasonal: bool) -> str:
    return f"{noise}, {'seasonal' if seasonal else 'no seasonal'}"


def summarise(hourly: pd.DataFrame) -> pd.Series:
    errors = hourly["forecast_error_c"]
    actual = hourly["actual_temperature_c"]
    return pd.Series(
        {
            "mae_c": errors.abs().mean(),
            "rmse_c": np.sqrt((errors**2).mean()),
            "bias_c": errors.mean(),
            "crps_c": hourly["crps_c"].mean(),
            "coverage_50": (
                (actual >= hourly["p25_c"]) & (actual <= hourly["p75_c"])
            ).mean(),
            "coverage_90": (
                (actual >= hourly["p05_c"]) & (actual <= hourly["p95_c"])
            ).mean(),
            "width_90_c": (hourly["p95_c"] - hourly["p05_c"]).mean(),
        }
    )


def compare_variants(
    data_file: Path = DATA_FILE,
    output_dir: Path | None = None,
    ensemble_size: int = ENSEMBLE_SIZE,
    block_hours: int = BLOCK_HOURS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    temperatures = load_temperatures(data_file)
    seasonal_coefficients = fit_seasonal_baseline(load_baseline_temperatures())
    year = int(temperatures.index[0].year)
    output_dir = output_dir or RESULTS_DIR
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    rows: list[pd.DataFrame] = []
    skipped = 0
    for forecast_date in dates:
        if forecast_skip_reason(temperatures, forecast_date) is not None:
            skipped += 1
            continue
        date_label = forecast_date.strftime("%Y-%m-%d")
        for noise, seasonal in VARIANTS:
            forecast_times, ensemble = predict_temperature_ensemble(
                temperatures,
                date_label,
                ensemble_size=ensemble_size,
                seed=seed + forecast_date.dayofyear,
                seasonal_coefficients=seasonal_coefficients if seasonal else None,
                noise=noise.lower(),
                block_hours=block_hours,
            )
            actual = temperatures.reindex(forecast_times)
            valid = actual.notna().to_numpy()
            if not valid.any():
                continue
            ensemble = ensemble[:, valid]
            observed = actual.to_numpy(dtype=float)[valid]
            p05, p25, p75, p95 = np.percentile(ensemble, [5, 25, 75, 95], axis=0)
            mean = ensemble.mean(axis=0)
            rows.append(
                pd.DataFrame(
                    {
                        "date": date_label,
                        "variant": variant_name(noise, seasonal),
                        "lead_hours": np.arange(1, 25)[valid],
                        "actual_temperature_c": observed,
                        "forecast_mean_c": mean,
                        "p05_c": p05,
                        "p25_c": p25,
                        "p75_c": p75,
                        "p95_c": p95,
                        "forecast_error_c": mean - observed,
                        "crps_c": ensemble_crps(ensemble, observed),
                    },
                    index=pd.Index(forecast_times[valid], name="timestamp_utc"),
                )
            )

    hourly = pd.concat(rows)
    names = [variant_name(*variant) for variant in VARIANTS]
    summary = hourly.groupby("variant").apply(summarise, include_groups=False).loc[names]

    # Paired day-block bootstrap of the CRPS difference to the reference.
    daily_crps = hourly.groupby(["variant", "date"])["crps_c"].mean().unstack(0)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(daily_crps), (5000, len(daily_crps)))
    for name in names:
        difference = (daily_crps[name] - daily_crps[REFERENCE]).to_numpy()
        means = difference[resamples].mean(axis=1)
        summary.loc[name, "crps_vs_reference_c"] = difference.mean()
        summary.loc[name, "crps_vs_reference_ci_low"] = np.percentile(means, 2.5)
        summary.loc[name, "crps_vs_reference_ci_high"] = np.percentile(means, 97.5)

    hourly["lead_bin"] = pd.cut(hourly["lead_hours"], LEAD_BINS).astype(str)
    by_lead = (
        hourly.groupby(["lead_bin", "variant"], sort=False)
        .apply(summarise, include_groups=False)[["crps_c", "coverage_90", "width_90_c"]]
        .unstack("variant")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_file = output_dir / f"langevin_variants_hourly_{year}.csv"
    summary_file = output_dir / f"langevin_variants_summary_{year}.csv"
    lead_file = output_dir / f"langevin_variants_by_lead_{year}.csv"
    plot_file = output_dir / f"langevin_variants_{year}.png"
    hourly.drop(columns="lead_bin").to_csv(hourly_file, float_format="%.4f")
    summary.to_csv(summary_file, float_format="%.4f")
    by_lead.to_csv(lead_file, float_format="%.4f")

    per_lead = hourly.groupby(["lead_hours", "variant"]).apply(summarise, include_groups=False)
    figure, (coverage_axis, crps_axis) = plt.subplots(1, 2, figsize=(13, 5))
    for name in names:
        lead_values = per_lead.xs(name, level="variant")
        coverage_axis.plot(lead_values.index, lead_values["coverage_90"], label=name)
        crps_axis.plot(lead_values.index, lead_values["crps_c"], label=name)
    coverage_axis.axhline(0.9, color="black", linestyle="--", label="Target (90%)")
    coverage_axis.set(
        xlabel="Lead time [h]",
        ylabel="Share of actuals inside 5th-95th percentile",
        title="90% interval coverage",
    )
    crps_axis.set(xlabel="Lead time [h]", ylabel="CRPS [°C]", title="CRPS (lower is better)")
    coverage_axis.legend()
    crps_axis.legend()
    figure.suptitle(f"Langevin variants, walk-forward {year} (block length {block_hours} h)")
    figure.tight_layout()
    figure.savefig(plot_file, dpi=150)
    plt.close(figure)

    days = hourly["date"].nunique()
    print(f"Walk-forward comparison of Langevin variants for {year}")
    print(
        f"Evaluated {days} days ({len(hourly) // len(VARIANTS)} hours), skipped {skipped}; "
        f"{ensemble_size} members, block length {block_hours} h."
    )
    print(summary.round(3).to_string())
    print("\nBy lead time:")
    print(by_lead.round(3).to_string())
    print(f"\nSaved hourly results to {hourly_file}")
    print(f"Saved summary to {summary_file}")
    print(f"Saved lead-time breakdown to {lead_file}")
    print(f"Saved plot to {plot_file}")
    return hourly, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Gaussian and block-bootstrap noise in the Langevin model, "
            "each with and without the seasonal forcing."
        )
    )
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/).",
    )
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument(
        "--block-hours",
        type=int,
        default=BLOCK_HOURS,
        help=f"Block length for the block noise, a divisor of 24 (default: {BLOCK_HOURS}).",
    )
    args = parser.parse_args()
    compare_variants(args.data, args.output_dir, args.ensemble_size, args.block_hours)


if __name__ == "__main__":
    main()
