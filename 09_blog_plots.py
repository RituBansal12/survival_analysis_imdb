#!/usr/bin/env python3
"""
Blog-ready visualizations for the IMDb TV series survival analysis.
Generates polished figures suitable for publication.

Inputs:
  - data/tvseries_survival_clean.csv

Outputs:
  - visualizations/survival/km_by_genre_blog.png
  - visualizations/survival/hazard_shape_blog.png
  - visualizations/cox/cox_forest_blog.png
  - visualizations/parametric/overlay_blog.png

Usage:
  python 09_blog_plots.py --input data/tvseries_survival_clean.csv --top-genres 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines import ExponentialFitter, WeibullFitter, LogNormalFitter, LogLogisticFitter

sns.set_theme(style="whitegrid", context="talk")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["duration", "event"]).copy()
    df = df[(df["duration"] > 0) & (df["event"].isin([0, 1]))]
    return df


def top_genres(df: pd.DataFrame, k: int) -> List[str]:
    melted = pd.melt(
        df,
        id_vars=["duration", "event"],
        value_vars=[c for c in ["genre_1", "genre_2", "genre_3"] if c in df.columns],
        value_name="genre",
    )
    vc = (
        melted.dropna(subset=["genre"]).groupby("genre").size().sort_values(ascending=False)
    )
    return vc.head(k).index.astype(str).tolist()


def plot_km_by_genre_blog(df: pd.DataFrame, out_png: Path, top_k: int) -> None:
    tg = top_genres(df, top_k)
    # Assign first matching top genre
    def assign(row) -> str | None:
        for g in [row.get("genre_1"), row.get("genre_2"), row.get("genre_3")]:
            if pd.notna(g) and str(g) in tg:
                return str(g)
        return None

    groups = df.apply(assign, axis=1)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("Set2", n_colors=len(tg))
    for (name, color) in zip(sorted(set(groups.dropna())), colors):
        idx = groups == name
        if idx.sum() < 60:
            continue
        kmf = KaplanMeierFitter(label=str(name))
        kmf.fit(df.loc[idx, "duration"], event_observed=df.loc[idx, "event"])
        kmf.plot(ci_show=False, color=color, linewidth=2.5)
    plt.xlabel("Years since start")
    plt.ylabel("Survival S(t)")
    plt.title("Survival by Top Genres (Kaplan–Meier)")
    plt.legend(title="Genre", ncol=2, frameon=True)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_hazard_shape_blog(df: pd.DataFrame, out_png: Path) -> None:
    naf = NelsonAalenFitter(label="Overall")
    naf.fit(df["duration"], event_observed=df["event"])
    plt.figure(figsize=(10, 7))
    naf.plot(ci_show=False, color="#E45756", linewidth=2.5)
    plt.xlabel("Years since start")
    plt.ylabel("Cumulative hazard H(t)")
    plt.title("How Hazard Accumulates Over Time")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_cox_forest_blog(cox_summary_csv: Path, out_png: Path) -> None:
    # Expect columns: variable, HR, HR_lower95, HR_upper95, p
    if not cox_summary_csv.exists():
        print(f"[warn] Cox summary not found at {cox_summary_csv}; skipping forest plot.")
        return
    df = pd.read_csv(cox_summary_csv)
    # Sort and prettify labels
    df = df.sort_values("HR").reset_index(drop=True)
    labels = df["variable"].str.replace("_", " ").str.replace("genre  ", "genre:").str.replace("dec ", "decade:")

    plt.figure(figsize=(10, max(6, 0.45 * len(df) + 2)))
    y = np.arange(len(df))
    plt.hlines(y, df["HR_lower95"], df["HR_upper95"], color="#4C78A8", linewidth=2)
    plt.scatter(df["HR"], y, color="#000000", s=28)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, labels)
    plt.xscale("log")
    plt.xlabel("Hazard Ratio (log scale)")
    plt.title("Cox PH: Adjusted Hazard Ratios")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_parametric_overlay_blog(df: pd.DataFrame, out_png: Path) -> None:
    T = df["duration"].values
    E = df["event"].values
    kmf = KaplanMeierFitter(label="KM")
    kmf.fit(T, event_observed=E)

    models = [
        ("Exponential", ExponentialFitter()),
        ("Weibull", WeibullFitter()),
        ("LogNormal", LogNormalFitter()),
        ("LogLogistic", LogLogisticFitter()),
    ]

    fits = []
    for name, m in models:
        try:
            m.fit(T, event_observed=E)
            fits.append((name, m))
        except Exception as e:
            print(f"[warn] {name} fit failed: {e}")

    t_grid = np.linspace(0, float(np.nanpercentile(T, 99.5)), 250)

    plt.figure(figsize=(10, 7))
    kmf.plot(ci_show=False, color="black", linewidth=2.5)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, m), c in zip(fits, colors):
        try:
            sf = m.survival_function_at_times(t_grid)
            plt.plot(t_grid, sf.values, label=name, color=c, linewidth=2.2)
        except Exception:
            try:
                sf_df = m.survival_function_
                plt.plot(sf_df.index.values, sf_df.iloc[:, 0].values, label=name, color=c, linewidth=2.2)
            except Exception:
                pass

    plt.xlabel("Years since start")
    plt.ylabel("Survival S(t)")
    plt.title("Parametric Fits vs Kaplan–Meier")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blog-ready visualizations")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--cox-summary", type=Path, default=Path("results/cox_summary.csv"))
    p.add_argument("--top-genres", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)

    plot_km_by_genre_blog(df, Path("visualizations/survival/km_by_genre_blog.png"), args.top_genres)
    plot_hazard_shape_blog(df, Path("visualizations/survival/hazard_shape_blog.png"))
    plot_cox_forest_blog(args.cox_summary, Path("visualizations/cox/cox_forest_blog.png"))
    plot_parametric_overlay_blog(df, Path("visualizations/parametric/overlay_blog.png"))

    print("Saved blog-ready plots to visualizations/survival, cox, and parametric folders.")


if __name__ == "__main__":
    main()
