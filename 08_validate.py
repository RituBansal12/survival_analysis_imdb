#!/usr/bin/env python3
"""
Validation & robustness checks for survival models.

Inputs:
  - data/tvseries_survival_clean.csv

Outputs:
  - results/validation_metrics.json
  - visualizations/validation/calibration_t0.png
  - visualizations/validation/bootstrap_cindex_hist.png

Usage:
  python 08_validate.py \
    --input data/tvseries_survival_clean.csv \
    --metrics results/validation_metrics.json \
    --vizdir visualizations/validation \
    --test-size 0.2 \
    --seed 42 \
    --top-genres 10 \
    --t0 5.0 \
    --bootstrap 100
"""
from __future__ import annotations

import argparse
import json
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
from lifelines.utils import concordance_index

sns.set_theme(style="whitegrid")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_decade(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return ((x // 10) * 10).astype("Int64")


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
    # Numeric
    X["averageRating"] = pd.to_numeric(X.get("averageRating"), errors="coerce")
    X["numVotes"] = pd.to_numeric(X.get("numVotes"), errors="coerce")
    X["numEpisodes"] = pd.to_numeric(X.get("numEpisodes"), errors="coerce")

    # logs
    X["log_numVotes"] = np.log1p(X["numVotes"]).astype(float)
    X["log_numEpisodes"] = np.log1p(X["numEpisodes"]).astype(float)

    # start decade dummies
    X["startDecade"] = to_decade(X.get("startYear"))
    X = pd.get_dummies(X, columns=["startDecade"], prefix="dec", drop_first=True, dummy_na=False)

    # top genres one-hot
    tg = top_genres(X, top_k_genres)
    for g in tg:
        mask = False
        for col in ["genre_1", "genre_2", "genre_3"]:
            if col in X.columns:
                mask = mask | (X[col].astype(str) == g)
        X[f"genre__{g}"] = mask.astype(int)

    covariates = [
        "averageRating",
        "log_numVotes",
        "log_numEpisodes",
    ] + [c for c in X.columns if c.startswith("dec_") or c.startswith("genre__")]

    X = X[["duration", "event"] + covariates].copy()
    X = X.dropna(subset=["duration", "event"]).copy()
    for c in covariates:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    X = X[(X["duration"] > 0) & (X["event"].isin([0, 1]))]
    return X


def split_train_test(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_n = int(round(n * test_size))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def fit_cox(X: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
    cph.fit(X, duration_col="duration", event_col="event", show_progress=False)
    return cph


def compute_cindex(cph: CoxPHFitter, X: pd.DataFrame) -> float:
    # Use negative risk score so that higher risk => shorter time
    risk = np.squeeze(cph.predict_partial_hazard(X).values)
    return float(concordance_index(X["duration"].values, -risk, X["event"].values))


def calibration_plot(cph: CoxPHFitter, X: pd.DataFrame, t0: float, out_png: Path) -> float:
    # Predict survival at t0
    sf = cph.predict_survival_function(X, times=[t0])  # 1 x n
    pred = sf.iloc[0, :].values  # predicted S(t0)
    # Bin into deciles
    q = pd.qcut(pred, q=10, duplicates="drop")
    groups = pd.Series(q, index=X.index)

    # Observed survival at t0 per decile via KM
    obs = []
    mid = []
    kmf = KaplanMeierFitter()
    for lvl, idx in groups.groupby(groups, observed=False).groups.items():
        sub = X.loc[idx]
        if len(sub) < 30:
            continue
        kmf.fit(sub["duration"], event_observed=sub["event"])
        # interpolate survival at t0
        # lifelines survival_function_ at fitted timeline; use nearest index value
        t_idx = np.searchsorted(kmf.survival_function_.index.values, t0, side="right") - 1
        if t_idx < 0:
            s_obs = 1.0
        else:
            s_obs = float(kmf.survival_function_.iloc[t_idx, 0])
        obs.append(s_obs)
        mid.append(float(sub.shape[0]))

    # For plotting, recompute bin centers and mean predicted/observed
    df_plot = (
        pd.DataFrame({"pred": pred, "group": groups})
        .dropna()
        .groupby("group", observed=False)
        .agg(pred_mean=("pred", "mean"), n=("pred", "size"))
        .reset_index(drop=True)
    )
    # Align observed and predicted by order
    df_plot["obs"] = obs[: len(df_plot)]

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_plot, x="pred_mean", y="obs", s=60, color="#4C78A8")
    lims = [0, 1]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel(f"Predicted S({t0:.1f})")
    plt.ylabel(f"Observed S({t0:.1f}) (KM)")
    plt.title("Calibration plot at time horizon")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Simple calibration error (RMSE)
    rmse = float(np.sqrt(np.mean((df_plot["pred_mean"] - df_plot["obs"]) ** 2)))
    return rmse


def bootstrap_cindex(cph_base: CoxPHFitter, X_train: pd.DataFrame, X_test: pd.DataFrame, B: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(X_train)
    vals: list[float] = []
    for b in range(B):
        try:
            idx = rng.integers(0, n, size=n)
            boot = X_train.iloc[idx]
            cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
            cph.fit(boot, duration_col="duration", event_col="event", show_progress=False)
            vals.append(compute_cindex(cph, X_test))
        except Exception:
            continue
    return vals


def sensitivity_outlier_refit(X_train: pd.DataFrame, X_test: pd.DataFrame, q: float = 0.99) -> float:
    thr = float(X_train["duration"].quantile(q))
    train2 = X_train[X_train["duration"] <= thr]
    if len(train2) < 100:
        return np.nan
    cph2 = fit_cox(train2)
    return compute_cindex(cph2, X_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation & robustness")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--metrics", type=Path, default=Path("results/validation_metrics.json"))
    p.add_argument("--vizdir", type=Path, default=Path("visualizations/validation"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-genres", type=int, default=10)
    p.add_argument("--t0", type=float, default=5.0)
    p.add_argument("--bootstrap", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    X = build_design_matrix(raw, args.top_genres)
    train, test = split_train_test(X, args.test_size, args.seed)

    cph = fit_cox(train)
    c_train = compute_cindex(cph, train)
    c_test = compute_cindex(cph, test)

    # Calibration plot at t0 on test set
    calib_rmse = calibration_plot(cph, test, args.t0, Path(args.vizdir) / "calibration_t0.png")

    # Bootstrap distribution of test c-index
    vals = bootstrap_cindex(cph, train, test, args.bootstrap, args.seed)
    if vals:
        lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        mean_val, std_val = float(np.mean(vals)), float(np.std(vals))
    else:
        lo = hi = mean_val = std_val = float("nan")

    # Plot histogram
    if vals:
        plt.figure(figsize=(7, 5))
        sns.histplot(vals, bins=20, kde=True, color="#72B7B2")
        plt.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1)
        plt.xlabel("C-index (test)")
        plt.ylabel("Count")
        plt.title("Bootstrap distribution of C-index on test")
        plt.tight_layout()
        ensure_dir(Path(args.vizdir))
        plt.savefig(Path(args.vizdir) / "bootstrap_cindex_hist.png", dpi=150)
        plt.close()

    # Sensitivity: remove extreme outliers (top 1% duration) and re-fit
    sens_c = sensitivity_outlier_refit(train, test, q=0.99)

    metrics = {
        "c_index_train": c_train,
        "c_index_test": c_test,
        "calibration_rmse_t0": calib_rmse,
        "bootstrap_mean": mean_val,
        "bootstrap_std": std_val,
        "bootstrap_ci95": [lo, hi],
        "sensitivity_c_index_test": sens_c,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }

    ensure_dir(Path(args.metrics).parent)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {args.metrics} and plots to {args.vizdir}")


if __name__ == "__main__":
    main()
