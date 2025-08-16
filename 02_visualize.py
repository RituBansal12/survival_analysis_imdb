#!/usr/bin/env python3
"""
Generate exploratory visualizations for the TV series survival dataset.

Input:
  - CSV: data/tvseries_survival.csv

Output:
  - 8 PNG figures saved in visualizations/ 

Usage:
  python 02_visualize.py --input data/tvseries_survival.csv --outdir visualizations
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _to_decade(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    decade = (s // 10) * 10
    return decade.astype("Int64")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    expected = {
        "tconst",
        "title",
        "startYear",
        "endYear",
        "duration",
        "event",
        "genres",
        "genre_1",
        "genre_2",
        "genre_3",
        "averageRating",
        "numVotes",
        "numEpisodes",
        "maxSeason",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input CSV: {missing}")

    # Helpful derived columns
    df["startDecade"] = _to_decade(df["startYear"])
    df["endDecade"] = _to_decade(df["endYear"])  # may be NA for ongoing
    # Cast for safety
    for c, dtype in [
        ("duration", "float"),
        ("event", "int"),
        ("averageRating", "float"),
        ("numVotes", "float"),
        ("numEpisodes", "float"),
        ("maxSeason", "float"),
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# 1. Lifespan of Shows — Histogram of duration

def plot_lifespan_hist(df: pd.DataFrame, outpath: Path) -> None:
    data = df["duration"].dropna()
    plt.figure(figsize=(9, 6))
    sns.histplot(data, bins=60, kde=True, color="#4C78A8")
    plt.xlabel("Duration (years)")
    plt.ylabel("Count")
    plt.title("TV Show Lifespan Distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 2. Events — Bar chart of ended vs ongoing proportion

def plot_event_proportions(df: pd.DataFrame, outpath: Path) -> None:
    # event: 1 ended, 0 ongoing
    counts = df["event"].value_counts(dropna=False).rename(index={0: "Ongoing", 1: "Ended"})
    order = ["Ongoing", "Ended"]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.loc[order].index, y=(counts.loc[order] / counts.sum()) * 100, palette=["#72B7B2", "#F58518"])
    plt.ylabel("Percentage of shows (%)")
    plt.xlabel("")
    plt.title("Proportion of Shows: Ongoing vs Ended")
    for i, val in enumerate((counts.loc[order] / counts.sum()) * 100):
        plt.text(i, val + 1, f"{val:.1f}%", ha="center")
    plt.ylim(0, max(((counts.loc[order] / counts.sum()) * 100).max() + 8, 30))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 3. Trends Over Time — Median duration by startYear decade

def plot_median_duration_by_start_decade(df: pd.DataFrame, outpath: Path) -> None:
    tmp = df.dropna(subset=["startDecade", "duration"]).copy()
    g = tmp.groupby("startDecade")["duration"].median().reset_index()
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=g, x="startDecade", y="duration", marker="o", color="#E45756")
    plt.xlabel("Start decade")
    plt.ylabel("Median duration (years)")
    plt.title("Median Show Duration by Start Decade")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 4. Genre Exploration — Median duration by top 10 genres

def plot_median_duration_top_genres(df: pd.DataFrame, outpath: Path, top_n: int = 10) -> None:
    # Melt genre_1..3 into single column
    melted = pd.melt(
        df,
        id_vars=["duration"],
        value_vars=["genre_1", "genre_2", "genre_3"],
        value_name="genre",
    ).dropna(subset=["genre", "duration"])  # drop NA genres and duration
    # Most frequent genres
    top_genres = (
        melted["genre"].value_counts().head(top_n).index.tolist()
    )
    subset = melted[melted["genre"].isin(top_genres)].copy()
    stats = subset.groupby("genre")["duration"].median().sort_values(ascending=False).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=stats, y="genre", x="duration", palette="viridis")
    plt.xlabel("Median duration (years)")
    plt.ylabel("Genre")
    plt.title("Median Duration by Top Genres")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 5. Ratings and Popularity — Scatterplot of averageRating vs duration

def plot_rating_vs_duration(df: pd.DataFrame, outpath: Path) -> None:
    tmp = df.dropna(subset=["averageRating", "duration"]).copy()
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=tmp, x="averageRating", y="duration", alpha=0.2, edgecolor=None, color="#54A24B")
    sns.regplot(data=tmp, x="averageRating", y="duration", scatter=False, color="#000000", line_kws={"linewidth": 1})
    plt.xlabel("Average rating")
    plt.ylabel("Duration (years)")
    plt.title("Do Higher-Rated Shows Last Longer?")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 6. Episodes and Seasons — Histogram of numEpisodes

def plot_numepisodes_hist(df: pd.DataFrame, outpath: Path) -> None:
    tmp = df["numEpisodes"].dropna()
    plt.figure(figsize=(9, 6))
    sns.histplot(tmp, bins=60, color="#4C78A8")
    plt.yscale("log")  # skewed distribution
    plt.xlabel("Number of episodes")
    plt.ylabel("Count (log scale)")
    plt.title("Distribution of Episode Counts")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 7. Multi-dimensional — Correlation heatmap of numeric variables

def plot_correlation_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    cols = ["duration", "averageRating", "numVotes", "numEpisodes", "maxSeason", "event"]
    tmp = df[cols].copy()
    corr = tmp.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# 8. Survival-Oriented EDA — Cumulative ended shows by decade (based on endYear)

def plot_cumulative_endings(df: pd.DataFrame, outpath: Path) -> None:
    ended = df[(df["event"] == 1) & df["endDecade"].notna()].copy()
    g = ended.groupby("endDecade").size().rename("ended_count").reset_index()
    g = g.sort_values("endDecade")
    g["cumulative_ended"] = g["ended_count"].cumsum()

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=g, x="endDecade", y="cumulative_ended", marker="o", color="#F58518")
    plt.xlabel("End decade")
    plt.ylabel("Cumulative number of ended shows")
    plt.title("Cumulative Ended Shows Over Time")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_all(df: pd.DataFrame, outdir: Path) -> None:
    _ensure_outdir(outdir)
    outputs = [
        (plot_lifespan_hist, outdir / "01_lifespan_histogram.png"),
        (plot_event_proportions, outdir / "02_event_proportions.png"),
        (plot_median_duration_by_start_decade, outdir / "03_median_duration_by_start_decade.png"),
        (plot_median_duration_top_genres, outdir / "04_median_duration_top_genres.png"),
        (plot_rating_vs_duration, outdir / "05_rating_vs_duration.png"),
        (plot_numepisodes_hist, outdir / "06_numepisodes_histogram.png"),
        (plot_correlation_heatmap, outdir / "07_correlation_heatmap.png"),
        (plot_cumulative_endings, outdir / "08_cumulative_endings_by_decade.png"),
    ]
    for func, path in outputs:
        print(f"[plot] {path.name} ...")
        if func is plot_median_duration_top_genres:
            func(df, path, top_n=10)
        else:
            func(df, path)
    print(f"Saved {len(outputs)} figures to {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate visualization figures")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival.csv"), help="Input CSV path")
    p.add_argument("--outdir", type=Path, default=Path("visualizations"), help="Output directory for figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    run_all(df, args.outdir)


if __name__ == "__main__":
    main()
