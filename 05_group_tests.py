#!/usr/bin/env python3
"""
Group comparisons with Logrank tests.

Inputs:
  - data/tvseries_survival_clean.csv

Outputs:
  - results/logrank_tests.csv
  - results/logrank_significant_pairs.csv
  - visualizations/survival/logrank_pvalues_heatmap.png

Usage:
  python 05_group_tests.py \
    --input data/tvseries_survival_clean.csv \
    --out results/logrank_tests.csv \
    --heatmap visualizations/survival/logrank_pvalues_heatmap.png \
    --top-genres 8
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import List

# Non-interactive backend for safety
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines.statistics import logrank_test

sns.set_theme(style="whitegrid")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def group_membership(df: pd.DataFrame, label: str) -> pd.Series:
    # A show belongs to genre if any of the three genre columns equals the label
    cols = [c for c in ["genre_1", "genre_2", "genre_3"] if c in df.columns]
    m = False
    for c in cols:
        m = m | (df[c].astype(str) == label)
    return m


def run_logrank(df: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series) -> tuple[float, float, int, int]:
    a = df.loc[mask_a]
    b = df.loc[mask_b]
    if len(a) < 20 or len(b) < 20:
        return (np.nan, np.nan, len(a), len(b))
    res = logrank_test(a["duration"], b["duration"], event_observed_A=a["event"], event_observed_B=b["event"])
    return (float(res.test_statistic), float(res.p_value), len(a), len(b))


def rating_groups(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    q = x.quantile([0.33, 0.66])
    bins = [-np.inf, q.iloc[0], q.iloc[1], np.inf]
    labels = ["Low", "Medium", "High"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)


def to_decade(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return ((x // 10) * 10).astype("Int64")


def compute_tests(df: pd.DataFrame, top_k_genres: int) -> pd.DataFrame:
    rows = []

    # Genres
    tg = top_genres(df, top_k_genres)
    for a, b in combinations(tg, 2):
        ma = group_membership(df, a)
        mb = group_membership(df, b)
        stat, p, na, nb = run_logrank(df, ma, mb)
        rows.append({
            "grouping": "genre",
            "group_a": a,
            "group_b": b,
            "n_a": na,
            "n_b": nb,
            "test": "logrank",
            "statistic": stat,
            "p_value": p,
        })

    # Rating groups
    rg = rating_groups(df["averageRating"]).astype("string")
    levels = [lvl for lvl in ["Low", "Medium", "High"] if (rg == lvl).sum() >= 20]
    for a, b in combinations(levels, 2):
        ma = rg == a
        mb = rg == b
        stat, p, na, nb = run_logrank(df, ma, mb)
        rows.append({
            "grouping": "rating_group",
            "group_a": a,
            "group_b": b,
            "n_a": na,
            "n_b": nb,
            "test": "logrank",
            "statistic": stat,
            "p_value": p,
        })

    # Start decade
    dec = to_decade(df["startYear"]).astype("Int64")
    vals = [str(v) for v, cnt in dec.value_counts().items() if cnt >= 50]
    for a, b in combinations(vals, 2):
        ma = (dec.astype(str) == a)
        mb = (dec.astype(str) == b)
        stat, p, na, nb = run_logrank(df, ma, mb)
        rows.append({
            "grouping": "start_decade",
            "group_a": a,
            "group_b": b,
            "n_a": na,
            "n_b": nb,
            "test": "logrank",
            "statistic": stat,
            "p_value": p,
        })

    return pd.DataFrame(rows)


def plot_heatmap_for_genres(results: pd.DataFrame, outpath: Path) -> None:
    df = results[results["grouping"] == "genre"].copy()
    if df.empty:
        return
    labels = sorted(set(df["group_a"]).union(set(df["group_b"])))
    mat = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, r in df.iterrows():
        mat.loc[r["group_a"], r["group_b"]] = r["p_value"]
        mat.loc[r["group_b"], r["group_a"]] = r["p_value"]
    np.fill_diagonal(mat.values, 0.0)

    plt.figure(figsize=(0.9*len(labels)+4, 0.9*len(labels)+3))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "p-value"})
    plt.title("Logrank Test p-values between Top Genres")
    plt.tight_layout()
    ensure_parent(outpath)
    plt.savefig(outpath, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Logrank tests between groups")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--out", type=Path, default=Path("results/logrank_tests.csv"))
    p.add_argument("--significant-out", type=Path, default=Path("results/logrank_significant_pairs.csv"))
    p.add_argument("--heatmap", type=Path, default=Path("visualizations/survival/logrank_pvalues_heatmap.png"))
    p.add_argument("--top-genres", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    res = compute_tests(df, args.top_genres)
    ensure_parent(args.out)
    res.to_csv(args.out, index=False)
    # Significant pairs (genre and rating_group only), sorted by p-value
    sig = res[res["grouping"].isin(["genre", "rating_group"])].copy()
    sig = sig.dropna(subset=["p_value"]).sort_values("p_value")
    # Conventional significance threshold
    sig = sig[sig["p_value"] < 0.05]
    ensure_parent(args.significant_out)
    sig.to_csv(args.significant_out, index=False)
    plot_heatmap_for_genres(res, args.heatmap)
    print(
        f"Wrote {len(res)} tests to {args.out}, "
        f"{len(sig)} significant pairs to {args.significant_out}, "
        f"and heatmap to {args.heatmap}"
    )


if __name__ == "__main__":
    main()
