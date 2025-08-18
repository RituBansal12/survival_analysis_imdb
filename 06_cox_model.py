#!/usr/bin/env python3
"""
Cox Proportional Hazards modeling with diagnostics and forest plot.

Inputs:
  - data/tvseries_survival_clean.csv

Outputs:
  - results/cox_summary.csv
  - visualizations/cox/cox_forest.png
  - visualizations/cox/schoenfeld_residuals.png
  - visualizations/cox/loglog_survival_rating_groups.png

Usage:
  python 06_cox_model.py \
    --input data/tvseries_survival_clean.csv \
    --results results/cox_summary.csv \
    --vizdir visualizations/cox \
    --top-genres 10
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
from lifelines import CoxPHFitter, KaplanMeierFitter

sns.set_theme(style="whitegrid")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_decade(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return ((x // 10) * 10).astype("Int64")


def rating_groups(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    q = x.quantile([0.33, 0.66])
    bins = [-np.inf, q.iloc[0], q.iloc[1], np.inf]
    labels = ["Low", "Medium", "High"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)


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


def build_design_matrix(df: pd.DataFrame, top_k_genres: int) -> pd.DataFrame:
    X = df.copy()
    # Base numeric covariates
    X["averageRating"] = pd.to_numeric(X.get("averageRating"), errors="coerce")
    X["numVotes"] = pd.to_numeric(X.get("numVotes"), errors="coerce")
    X["numEpisodes"] = pd.to_numeric(X.get("numEpisodes"), errors="coerce")

    # logs
    X["log_numVotes"] = np.log1p(X["numVotes"]).astype(float)
    X["log_numEpisodes"] = np.log1p(X["numEpisodes"]).astype(float)

    # start decade dummies
    X["startDecade"] = to_decade(X.get("startYear"))
    X = pd.get_dummies(X, columns=["startDecade"], prefix="dec", drop_first=True, dummy_na=False)

    # top genres dummies (one-vs-rest; presence in any of 3 genre columns)
    tg = top_genres(X, top_k_genres)
    for g in tg:
        mask = False
        for col in ["genre_1", "genre_2", "genre_3"]:
            if col in X.columns:
                mask = mask | (X[col].astype(str) == g)
        X[f"genre__{g}"] = mask.astype(int)

    # Keep relevant columns
    covariates = [
        "averageRating",
        "log_numVotes",
        "log_numEpisodes",
    ] + [c for c in X.columns if c.startswith("dec_") or c.startswith("genre__")]

    # Clean rows
    X = X[["duration", "event"] + covariates].copy()
    X = X.dropna(subset=["duration", "event"])  # time, event
    # For covariates, fill missing numeric with 0 (safe for dummies/logs)
    for c in covariates:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    # Ensure valid durations/events
    X = X[(X["duration"] > 0) & (X["event"].isin([0, 1]))]
    return X


def fit_cox(X: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    cph.fit(X, duration_col="duration", event_col="event", show_progress=False)
    return cph


def export_summary(cph: CoxPHFitter, out_csv: Path) -> pd.DataFrame:
    summ = cph.summary.reset_index()
    # lifelines names the index column 'covariate' (newer versions) or it defaults to 'index'
    if "covariate" in summ.columns:
        summ = summ.rename(columns={"covariate": "variable"})
    elif "index" in summ.columns:
        summ = summ.rename(columns={"index": "variable"})
    else:
        # Fallback: create variable from the original index
        summ["variable"] = cph.summary.index.astype(str).values

    # Compute HR and CI from coefficients
    summ["HR"] = np.exp(summ["coef"])
    summ["HR_lower95"] = np.exp(summ["coef"] - 1.96 * summ["se(coef)"])
    summ["HR_upper95"] = np.exp(summ["coef"] + 1.96 * summ["se(coef)"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summ[["variable", "HR", "HR_lower95", "HR_upper95", "p"]].to_csv(out_csv, index=False)
    return summ


def plot_forest(summ: pd.DataFrame, out_png: Path) -> None:
    df = summ[["variable", "HR", "HR_lower95", "HR_upper95", "p"]].copy()
    df = df.sort_values("HR").reset_index(drop=True)

    plt.figure(figsize=(8, max(5, 0.35 * len(df) + 2)))
    y = np.arange(len(df))
    plt.hlines(y, df["HR_lower95"], df["HR_upper95"], color="#4C78A8")
    plt.scatter(df["HR"], y, color="#E45756")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, df["variable"])
    plt.xscale("log")
    plt.xlabel("Hazard Ratio (log scale)")
    plt.title("Cox PH Hazard Ratios with 95% CI")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_loglog_km_by_rating(df: pd.DataFrame, out_png: Path) -> None:
    # log(-log(S)) by rating groups
    kmf = KaplanMeierFitter()
    df = df.dropna(subset=["duration", "event"]).copy()
    df = df[(df["duration"] > 0) & (df["event"].isin([0, 1]))]
    rg = rating_groups(df["averageRating"]).astype("string")

    plt.figure(figsize=(9, 6))
    for lvl in ["Low", "Medium", "High"]:
        idx = rg == lvl
        if idx.sum() < 30:
            continue
        kmf.fit(df.loc[idx, "duration"], event_observed=df.loc[idx, "event"], label=lvl)
        sf = kmf.survival_function_
        y = -np.log(-np.log(sf.iloc[:, 0].clip(lower=1e-6)))
        plt.plot(sf.index, y, label=lvl)
    plt.legend(title="Rating group")
    plt.xlabel("Time (years)")
    plt.ylabel("log(-log(S(t)))")
    plt.title("log(-log(S)) by Rating Group")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_schoenfeld_like(cph: CoxPHFitter, X: pd.DataFrame, out_png: Path, max_vars: int = 12) -> None:
    # Plot scaled Schoenfeld residuals vs time for top |coef| variables
    try:
        res = cph.compute_residuals(X, kind="schoenfeld")
    except Exception:
        return
    # Select variables present in residuals
    vars_in_model = [v for v in cph.params_.index if v in res.columns]
    order = np.argsort(np.abs(cph.params_.values))[::-1]
    vars_sorted = [cph.params_.index[i] for i in order if cph.params_.index[i] in vars_in_model][:max_vars]

    n = len(vars_sorted)
    if n == 0:
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(5*cols, 3.2*rows))
    t = res.index.values
    for i, var in enumerate(vars_sorted):
        ax = plt.subplot(rows, cols, i+1)
        y = res[var].values
        ax.scatter(t, y, s=6, alpha=0.5)
        # add lowess-like smoothing via rolling mean
        try:
            import pandas as _pd
            ys = _pd.Series(y).rolling(window=max(5, len(y)//20), min_periods=1, center=True).mean()
            ax.plot(t, ys, color="red", linewidth=1)
        except Exception:
            pass
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(var)
        ax.set_xlabel("Time")
        ax.set_ylabel("Schoenfeld")
    plt.suptitle("Scaled Schoenfeld residuals (diagnostic)")
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    X = build_design_matrix(df, args.top_genres)
    cph = fit_cox(X)

    # Export summary and forest
    summ = export_summary(cph, args.results)
    plot_forest(summ, Path(args.vizdir) / "cox_forest.png")

    # Diagnostics
    plot_schoenfeld_like(cph, X, Path(args.vizdir) / "schoenfeld_residuals.png")
    plot_loglog_km_by_rating(df, Path(args.vizdir) / "loglog_survival_rating_groups.png")

    print(f"Saved Cox summary to {args.results} and figures to {args.vizdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cox PH modeling")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--results", type=Path, default=Path("results/cox_summary.csv"))
    p.add_argument("--vizdir", type=Path, default=Path("visualizations/cox"))
    p.add_argument("--top-genres", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
