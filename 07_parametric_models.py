#!/usr/bin/env python3
"""
Parametric survival models: Exponential, Weibull, Log-Normal, Log-Logistic.
Compare AIC/BIC and overlay with KM curve.

Inputs:
  - data/tvseries_survival_clean.csv

Outputs:
  - results/parametric_comparison.csv
  - visualizations/parametric/overlay_overall.png

Usage:
  python 07_parametric_models.py \
    --input data/tvseries_survival_clean.csv \
    --out results/parametric_comparison.csv \
    --viz visualizations/parametric/overlay_overall.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import ExponentialFitter, WeibullFitter, LogNormalFitter, LogLogisticFitter


MODELS = [
    ("Exponential", ExponentialFitter),
    ("Weibull", WeibullFitter),
    ("LogNormal", LogNormalFitter),
    ("LogLogistic", LogLogisticFitter),
]


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["duration", "event"])  # basic
    df = df[(df["duration"] > 0) & (df["event"].isin([0, 1]))]
    return df


def fit_models(df: pd.DataFrame) -> list[tuple[str, object]]:
    fits: list[tuple[str, object]] = []
    T = df["duration"]
    E = df["event"]
    for name, Cls in MODELS:
        try:
            m = Cls()
            m.fit(T, event_observed=E)
            fits.append((name, m))
        except Exception as e:
            print(f"[warn] {name} fit failed: {e}")
    return fits


def collect_ic(fits: list[tuple[str, object]], n_events: int) -> pd.DataFrame:
    rows = []
    for name, m in fits:
        aic = float(getattr(m, "AIC_", np.nan))
        ll = float(getattr(m, "log_likelihood_", np.nan))
        k = len(getattr(m, "_fitted_parameter_names", []) or list(getattr(m, "_fitted_parameters", {}).keys()))
        bic = float(np.nan)
        if not np.isnan(ll) and k and n_events:
            bic = -2.0 * ll + k * np.log(n_events)
        rows.append({
            "model": name,
            "AIC": aic,
            "BIC": bic,
            "log_likelihood": ll,
            "k": k,
        })
    return pd.DataFrame(rows).sort_values(["AIC"]).reset_index(drop=True)


def plot_overlay(df: pd.DataFrame, fits: list[tuple[str, object]], outpath: Path) -> None:
    T = df["duration"].values
    E = df["event"].values
    t_grid = np.linspace(0, float(np.nanpercentile(T, 99.5)), 200)

    kmf = KaplanMeierFitter(label="KM")
    kmf.fit(T, event_observed=E)

    plt.figure(figsize=(9, 6))
    kmf.plot(ci_show=False, color="black")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, m), c in zip(fits, colors):
        try:
            sf = m.survival_function_at_times(t_grid)
            plt.plot(t_grid, sf.values, label=name, color=c)
        except Exception:
            # Fallback for lifelines versions without survival_function_at_times
            try:
                sf_df = m.survival_function_
                plt.plot(sf_df.index.values, sf_df.iloc[:, 0].values, label=name, color=c)
            except Exception:
                pass

    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability S(t)")
    plt.title("Parametric Fits vs Kaplan–Meier")
    plt.legend()
    plt.tight_layout()
    ensure_parent(outpath)
    plt.savefig(outpath, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parametric survival models")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--out", type=Path, default=Path("results/parametric_comparison.csv"))
    p.add_argument("--viz", type=Path, default=Path("visualizations/parametric/overlay_overall.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    fits = fit_models(df)
    ic = collect_ic(fits, int((df["event"] == 1).sum()))
    ensure_parent(args.out)
    ic.to_csv(args.out, index=False)
    plot_overlay(df, fits, args.viz)
    print(f"Saved model comparison to {args.out} and overlay to {args.viz}")


if __name__ == "__main__":
    main()
