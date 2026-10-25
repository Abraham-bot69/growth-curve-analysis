"""
Bacterial growth curve analysis (simulated data)

What this script does
1. Simulates OD600 readings for two incubation conditions (37 C and 30 C),
   three replicate wells each, read every 30 minutes for 24 hours.
2. Fits the modified Gompertz model (Zwietering et al., 1990) to each replicate.
3. Estimates lag time, maximum specific growth rate, doubling time and
   maximum OD for every replicate.
4. Compares the two conditions with a Welch t-test on growth rate.
5. Saves the data, the fit results and a figure.

All data here are SIMULATED. The point of the project is the method, which
works the same way on real plate-reader data.

Run:  python growth_curve_analysis.py
Needs: numpy, pandas, scipy, matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind

OUT = Path(__file__).parent
RNG = np.random.default_rng(42)  # fixed seed so results are reproducible

OD0 = 0.05  # starting OD600 of the culture
TIME = np.arange(0, 24.5, 0.5)  # hours, one reading every 30 minutes

# "True" settings used to simulate each condition.
# mu = max specific growth rate (per hour), lag = lag time (hours), od_max = plateau OD.
CONDITIONS = {
    "37 C": {"mu": 0.62, "lag": 1.5, "od_max": 1.60},
    "30 C": {"mu": 0.38, "lag": 2.5, "od_max": 1.40},
}
REPLICATES = 3


def gompertz(t, A, mu, lag):
    """Modified Gompertz model for y = ln(OD / OD0).

    A   : maximum value of y (the plateau)
    mu  : maximum specific growth rate (per hour)
    lag : lag time (hours)
    """
    return A * np.exp(-np.exp(mu * np.e / A * (lag - t) + 1))


def simulate():
    """Create a tidy table of simulated OD600 readings."""
    rows = []
    for cond, p in CONDITIONS.items():
        A_true = np.log(p["od_max"] / OD0)
        for rep in range(1, REPLICATES + 1):
            # Wells never behave identically, so jitter the settings a little.
            mu = p["mu"] * RNG.normal(1, 0.05)
            lag = p["lag"] * RNG.normal(1, 0.08)
            A = A_true * RNG.normal(1, 0.02)
            y = gompertz(TIME, A, mu, lag)
            od = OD0 * np.exp(y) * RNG.lognormal(0, 0.03, size=TIME.size)  # reading noise
            for t, v in zip(TIME, od):
                rows.append({"condition": cond, "replicate": rep, "time_h": t, "od600": round(v, 4)})
    return pd.DataFrame(rows)


def fit_replicates(df):
    """Fit the model to each replicate and collect the growth parameters."""
    results = []
    for (cond, rep), g in df.groupby(["condition", "replicate"]):
        t = g["time_h"].to_numpy()
        y = np.log(g["od600"].to_numpy() / g["od600"].iloc[0])  # ln(OD / OD at time 0)
        p0 = [y.max(), 0.4, 1.0]  # starting guesses
        popt, _ = curve_fit(gompertz, t, y, p0=p0, maxfev=10000)
        A, mu, lag = popt
        pred = gompertz(t, *popt)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        results.append({
            "condition": cond,
            "replicate": rep,
            "max_growth_rate_per_h": mu,
            "doubling_time_h": np.log(2) / mu,
            "lag_time_h": lag,
            "max_od600": g["od600"].iloc[0] * np.exp(A),
            "r_squared": r2,
            "A": A,
        })
    return pd.DataFrame(results)


def plot(df, fits, path):
    colors = {"37 C": "#c2410c", "30 C": "#0e7490"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: raw readings with fitted curves
    ax = axes[0]
    for cond, g in df.groupby("condition"):
        ax.scatter(g["time_h"], g["od600"], s=14, alpha=0.5, color=colors[cond])
        for _, r in fits[fits["condition"] == cond].iterrows():
            y = gompertz(TIME, r["A"], r["max_growth_rate_per_h"], r["lag_time_h"])
            od_start = g[(g["replicate"] == r["replicate"])]["od600"].iloc[0]
            ax.plot(TIME, od_start * np.exp(y), color=colors[cond], lw=1.4, alpha=0.9)
        ax.plot([], [], color=colors[cond], label=cond)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("OD600")
    ax.set_title("Simulated growth curves with Gompertz fits")
    ax.legend(title="Incubation")
    ax.grid(alpha=0.25)

    # Right: growth rate comparison
    ax = axes[1]
    for i, (cond, g) in enumerate(fits.groupby("condition")):
        vals = g["max_growth_rate_per_h"]
        ax.bar(i, vals.mean(), yerr=vals.std(ddof=1), capsize=6, color=colors[cond], alpha=0.8, width=0.55)
        ax.scatter([i] * len(vals), vals, color="black", s=22, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(sorted(fits["condition"].unique()))
    ax.set_ylabel("Max specific growth rate (per hour)")
    ax.set_title("Growth rate by condition (mean and SD, n = 3)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    df = simulate()
    df.to_csv(OUT / "simulated_od600.csv", index=False)

    fits = fit_replicates(df)
    fits.drop(columns="A").round(4).to_csv(OUT / "fit_results.csv", index=False)

    summary = (
        fits.groupby("condition")[["max_growth_rate_per_h", "doubling_time_h", "lag_time_h", "max_od600", "r_squared"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())

    a = fits[fits["condition"] == "37 C"]["max_growth_rate_per_h"]
    b = fits[fits["condition"] == "30 C"]["max_growth_rate_per_h"]
    t_stat, p_val = ttest_ind(a, b, equal_var=False)  # Welch t-test
    print(f"\nWelch t-test on growth rate, 37 C vs 30 C: t = {t_stat:.2f}, p = {p_val:.4f}")

    plot(df, fits, OUT / "growth_curves.png")
    print("Saved: simulated_od600.csv, fit_results.csv, growth_curves.png")


if __name__ == "__main__":
    main()
