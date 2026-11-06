"""Synthetic heat load samples from a logistic temperature-load relationship.

No real heat load data was available, so the load is generated from an
assumed logistic curve of outdoor temperature plus 1% Gaussian noise. Writes
data/synthetic_heat_load.csv and results/synthetic_heat_load.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOAD_MIN, LOAD_MAX = 5.0, 20.0
T_C, W = -5.0, 3.5  # curve midpoint [C] and width [C]
N, T_MIN, T_MAX = 14000, -35, 35  # number of samples and temperature range [C]
# 200 samples per °C. Forecast temperatures outside this range are clipped to
# its edge in predict_heat_load_ensemble, where the load curve is flat.
NOISE_SD = 0.01 * (LOAD_MAX - LOAD_MIN)
SEED = 0
# Seed of the reference ("observed") load built from observed temperatures.
REFERENCE_LOAD_SEED = 12345
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def heat_load(T, load_min=LOAD_MIN, load_max=LOAD_MAX, T_c=T_C, w=W):
    return load_min + (load_max - load_min) / (1 + np.exp((T - T_c) / w))


def simulate_true_load(
    temperatures: pd.Series, seed: int = REFERENCE_LOAD_SEED
) -> pd.Series:
    """Reference ("observed") heat load for observed temperatures.

    Generated the same way as the synthetic data, curve plus noise clipped to
    the load range, so it matches the data the forecast samples from. The
    noise-free curve would be unfair at the load floor, where clipping makes
    the synthetic loads average ~5.06 rather than 5.00. Pass the whole year so
    that every hour always gets the same noise draw.
    """
    rng = np.random.default_rng(seed)
    curve = heat_load(temperatures.to_numpy(dtype=float))
    noisy = np.clip(curve + rng.normal(0, NOISE_SD, len(curve)), LOAD_MIN, LOAD_MAX)
    return pd.Series(noisy, index=temperatures.index, name="reference_load")


def main():
    rng = np.random.default_rng(SEED)
    T = rng.uniform(T_MIN, T_MAX, N)
    load_true = heat_load(T)
    load = np.clip(load_true + rng.normal(0, NOISE_SD, N), LOAD_MIN, LOAD_MAX)

    df = pd.DataFrame({"temperature": T, "heat_load_true": load_true, "heat_load": load})
    df.to_csv(DATA_DIR / "synthetic_heat_load.csv", index=False)

    Tg = np.linspace(T_MIN, T_MAX, 300)
    plt.scatter(T, load, s=4, alpha=0.4, label="synthetic data")
    plt.plot(Tg, heat_load(Tg), "k", lw=2, label="true curve")
    plt.xlabel("Outdoor temperature [°C]")
    plt.ylabel("Heat load")
    plt.legend()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "synthetic_heat_load.png", dpi=150)


if __name__ == "__main__":
    main()