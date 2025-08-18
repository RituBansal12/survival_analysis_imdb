#!/usr/bin/env python3
"""
Exploratory Survival Analysis: Kaplan–Meier curves and hazards.

Inputs (default):
  - data/tvseries_survival_clean.csv

Outputs:
  - visualizations/survival/km_overall.png
  - visualizations/survival/km_by_genre.png
  - visualizations/survival/km_by_rating_group.png
  - visualizations/survival/km_by_start_decade.png
  - visualizations/survival/cumhaz_overall.png
  - visualizations/survival/nelson_aalen_overall.png
  - results/km_tables.csv

Usage:
  python 04_survival_eda.py \
    --input data/tvseries_survival_clean.csv \
    --vizdir visualizations/survival \
    --km-table results/km_tables.csv \
    --top-genres 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter

sns.set_theme(style="whitegrid")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce expected columns
    for c in ["duration", "event", "averageRating", "numVotes", "numEpisodes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_decade(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return ((x // 10) * 10).astype("Int64")


def rating_groups(s: pd.Series, labels=("Low", "Medium", "High")) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    q = x.quantile([0.33, 0.66])
    bins = [-np.inf, q.iloc[0], q.iloc[1], np.inf]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)


def top_genres(df: pd.DataFrame, k: int) -> list[str]:
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


def km_fit_table(kmf: KaplanMeierFitter, label: str) -> pd.DataFrame:
    t = kmf.survival_function_.reset_index().rename(columns={kmf.survival_function_.columns[0]: "S"})
    t["label"] = label
    return t


def plot_km_overall(df: pd.DataFrame, outpath: Path, tables: list[pd.DataFrame]) -> None:
    T, E = df["duration"], df["event"]
    kmf = KaplanMeierFitter(label="Overall")
    kmf.fit(T, event_observed=E)

    plt.figure(figsize=(9, 6))
    kmf.plot(ci_show=True, color="#1f77b4")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability S(t)")
    plt.title("Kaplan–Meier: Overall Survival")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    tables.append(km_fit_table(kmf, "Overall"))


def plot_km_by_category(df: pd.DataFrame, cat: pd.Series, outpath: Path, tables: list[pd.DataFrame], title: str) -> None:
    plt.figure(figsize=(10, 7))
    for name, idx in cat.dropna().groupby(cat.dropna()).groups.items():
        sub = df.loc[idx]
        if len(sub) < 30:
            continue
        kmf = KaplanMeierFitter(label=str(name))
        kmf.fit(sub["duration"], event_observed=sub["event"])
        kmf.plot(ci_show=False)
        tables.append(km_fit_table(kmf, str(name)))
    plt.xlabel("Time (years)")
    plt.ylabel("Survival probability S(t)")
    plt.title(title)
    plt.legend(title="Group", ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_cumhaz_overall(df: pd.DataFrame, outpath: Path) -> None:
    naf = NelsonAalenFitter(label="Overall")
    naf.fit(df["duration"], event_observed=df["event"])
    plt.figure(figsize=(9, 6))
    naf.plot()
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative hazard H(t)")
    plt.title("Cumulative Hazard (Nelson–Aalen)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def export_tables(tables: Iterable[pd.DataFrame], outpath: Path) -> None:
    if not tables:
        return
    df = pd.concat(tables, ignore_index=True)
    df = df.rename(columns={"timeline": "time"})
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)


def run(args: argparse.Namespace) -> None:
    df = load_data(args.input)
    # Sanity: drop rows with invalid duration/event
    df = df.dropna(subset=["duration", "event"]).copy()
    df = df[(df["duration"] > 0) & (df["event"].isin([0, 1]))]

    ensure_dir(args.vizdir)

    tables: list[pd.DataFrame] = []

    # Overall KM
    plot_km_overall(df, args.vizdir / "km_overall.png", tables)

    # By Genre (top K)
    tg = top_genres(df, args.top_genres)
    # Assign primary genre membership if present among top K (first match across three columns)
    def assign_top_genre(row) -> str | None:
        for g in [row.get("genre_1"), row.get("genre_2"), row.get("genre_3")]:
            if pd.notna(g) and str(g) in tg:
                return str(g)
        return None
    genre_series = df.apply(assign_top_genre, axis=1)
    plot_km_by_category(df, genre_series, args.vizdir / "km_by_genre.png", tables, "KM by Genre (Top)")

    # By rating group
    rgrp = rating_groups(df["averageRating"]).astype("string")
    plot_km_by_category(df, rgrp, args.vizdir / "km_by_rating_group.png", tables, "KM by Rating Group")

    # By start decade
    sdec = to_decade(df["startYear"]).astype("Int64").astype("string")
    plot_km_by_category(df, sdec, args.vizdir / "km_by_start_decade.png", tables, "KM by Start Decade")

    # Cumulative hazard overall
    plot_cumhaz_overall(df, args.vizdir / "cumhaz_overall.png")

    # Nelson–Aalen (same as cumhaz but save explicit name too)
    plot_cumhaz_overall(df, args.vizdir / "nelson_aalen_overall.png")

    # Export tables
    export_tables(tables, args.km_table)
    print(f"Saved KM plots to {args.vizdir} and tables to {args.km_table}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaplan–Meier EDA")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--vizdir", type=Path, default=Path("visualizations/survival"))
    p.add_argument("--km-table", type=Path, default=Path("results/km_tables.csv"))
    p.add_argument("--top-genres", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
