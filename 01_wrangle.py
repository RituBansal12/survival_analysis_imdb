#!/usr/bin/env python3
"""
Wrangle IMDb raw TSVs into a TV series survival analysis dataset.

Inputs (default paths assume repository layout):
- data/title.basics.tsv
- data/title.ratings.tsv
- data/title.episode.tsv

Output:
- data/tvseries_survival.csv

Steps:
1) Filter basics to tvSeries and keep key fields
2) Define survival variables: duration, event
3) Merge ratings
4) Aggregate episode info: numEpisodes, maxSeason
5) Clean types
6) Save csv

Usage:
  python 01_wrangle.py --data-dir data --output data/tvseries_survival.csv --current-year 2025

Note: Set current-year as the cutoff year (for ongoing shows)

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def read_tv_series_basics(basics_path: Path, current_year: int, chunksize: int = 500_000) -> pd.DataFrame:
    usecols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear",
        "endYear",
        "genres",
    ]
    tv_dfs = []
    # IMDb uses "\\N" for missing values
    na_vals = ["\\N"]

    for chunk in pd.read_csv(
        basics_path,
        sep="\t",
        usecols=usecols,
        dtype={
            "tconst": "string",
            "titleType": "string",
            "primaryTitle": "string",
            "startYear": "Int64",
            "endYear": "Int64",
            "genres": "string",
        },
        na_values=na_vals,
        quoting=3,  # csv.QUOTE_NONE
        chunksize=chunksize,
        low_memory=False,
    ):
        # Filter to tvSeries (case-insensitive to handle variations)
        mask_tv = chunk["titleType"].str.lower() == "tvseries"
        chunk = chunk.loc[mask_tv, [
            "tconst",
            "primaryTitle",
            "startYear",
            "endYear",
            "genres",
        ]].copy()

        # Drop missing startYear
        chunk = chunk[chunk["startYear"].notna()].copy()

        # Compute event and duration
        ended_mask = chunk["endYear"].notna()
        # event: 1 if ended, 0 if censored/ongoing
        chunk["event"] = ended_mask.astype("Int8")

        # duration
        # For ended shows: endYear - startYear; else: current_year - startYear
        dur = pd.Series(np.empty(len(chunk), dtype="float"), index=chunk.index)
        dur.loc[ended_mask] = (
            chunk.loc[ended_mask, "endYear"].astype("Int64")
            - chunk.loc[ended_mask, "startYear"].astype("Int64")
        ).astype("float")
        dur.loc[~ended_mask] = (
            current_year - chunk.loc[~ended_mask, "startYear"].astype("Int64")
        ).astype("float")
        # Ensure non-negative durations and drop invalid
        chunk["duration"] = dur
        chunk = chunk[chunk["duration"].notna() & (chunk["duration"] >= 0)].copy()

        # Rename and select final columns from basics
        chunk = chunk.rename(columns={"primaryTitle": "title"})[
            [
                "tconst",
                "title",
                "startYear",
                "endYear",
                "duration",
                "event",
                "genres",
            ]
        ]
        
        # Split genres into up to three columns (nullable)
        g = chunk["genres"].str.split(",", n=2, expand=True)
        chunk["genre_1"] = g[0]
        chunk["genre_2"] = g[1]
        chunk["genre_3"] = g[2]
        
        # Types
        chunk["title"] = chunk["title"].astype("string")
        chunk["genres"] = chunk["genres"].astype("string")
        chunk["genre_1"] = chunk["genre_1"].astype("string")
        chunk["genre_2"] = chunk["genre_2"].astype("string")
        chunk["genre_3"] = chunk["genre_3"].astype("string")
        chunk["duration"] = chunk["duration"].astype("Int64")
        chunk["event"] = chunk["event"].astype("Int8")

        tv_dfs.append(chunk)

    if not tv_dfs:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    tv = pd.concat(tv_dfs, ignore_index=True)
    return tv


def read_ratings(ratings_path: Path) -> pd.DataFrame:
    usecols = ["tconst", "averageRating", "numVotes"]
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=usecols,
        dtype={
            "tconst": "string",
            "averageRating": "float32",
            "numVotes": "Int64",
        },
        na_values=["\\N"],
        quoting=3,
        low_memory=False,
    )
    return ratings


def aggregate_episodes(episodes_path: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    # Accumulators
    counts: Dict[str, int] = {}
    max_season: Dict[str, int] = {}

    usecols = ["parentTconst", "seasonNumber"]
    for chunk in pd.read_csv(
        episodes_path,
        sep="\t",
        usecols=usecols,
        dtype={
            "parentTconst": "string",
            "seasonNumber": "Int64",
        },
        na_values=["\\N"],
        quoting=3,
        chunksize=chunksize,
        low_memory=False,
    ):
        # Drop rows with missing parentTconst
        chunk = chunk[chunk["parentTconst"].notna()].copy()

        # numEpisodes: simple count per parent
        cnt_series = chunk.groupby("parentTconst").size()
        for k, v in cnt_series.items():
            counts[k] = counts.get(k, 0) + int(v)

        # maxSeason: max of seasonNumber per parent, ignoring missing
        if "seasonNumber" in chunk.columns:
            # Drop NA seasonNumber for max computation
            non_na = chunk.dropna(subset=["seasonNumber"])
            if len(non_na):
                max_series = non_na.groupby("parentTconst")["seasonNumber"].max()
                for k, v in max_series.items():
                    vv = int(v)
                    if k in max_season:
                        if vv > max_season[k]:
                            max_season[k] = vv
                    else:
                        max_season[k] = vv

    # Build DataFrame
    if not counts:
        return pd.DataFrame(columns=["tconst", "numEpisodes", "maxSeason"])  # empty

    counts_s = pd.Series(counts, name="numEpisodes")
    max_season_s = pd.Series(max_season, name="maxSeason")
    agg = pd.concat([counts_s, max_season_s], axis=1)
    agg.index.name = "tconst"
    agg = agg.reset_index()
    # Types
    agg["numEpisodes"] = agg["numEpisodes"].astype("Int64")
    agg["maxSeason"] = agg["maxSeason"].astype("Int64")
    return agg


def build_dataset(data_dir: Path, output_path: Path, current_year: int) -> Path:
    basics_path = data_dir / "title.basics.tsv"
    ratings_path = data_dir / "title.ratings.tsv"
    episodes_path = data_dir / "title.episode.tsv"

    if not basics_path.exists() or not ratings_path.exists() or not episodes_path.exists():
        missing = [p.name for p in [basics_path, ratings_path, episodes_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required input files: {missing}")

    print("[1/5] Reading tv series from basics ...")
    tv = read_tv_series_basics(basics_path, current_year=current_year)
    print(f"  -> tv series rows: {len(tv):,}")

    print("[2/5] Reading ratings ...")
    ratings = read_ratings(ratings_path)
    print(f"  -> ratings rows: {len(ratings):,}")

    print("[3/5] Merging ratings ...")
    tv = tv.merge(ratings, on="tconst", how="left")

    print("[4/5] Aggregating episodes ...")
    episodes_agg = aggregate_episodes(episodes_path)
    print(f"  -> aggregated shows in episodes: {len(episodes_agg):,}")

    print("[5/5] Merging episode aggregates ...")
    tv = tv.merge(episodes_agg, on="tconst", how="left")

    # Final clean: enforce dtypes
    for col, dtype in [
        ("startYear", "Int64"),
        ("endYear", "Int64"),
        ("duration", "Int64"),
        ("event", "Int8"),
        ("averageRating", "float32"),
        ("numVotes", "Int64"),
        ("numEpisodes", "Int64"),
        ("maxSeason", "Int64"),
    ]:
        if col in tv.columns:
            tv[col] = tv[col].astype(dtype)

    # Save csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving CSV to {output_path} ...")
    tv.to_csv(output_path, index=False, encoding="utf-8")
    print("Done.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrangle IMDb TSVs into survival dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing IMDb TSV files")
    parser.add_argument("--output", type=Path, default=Path("data/tvseries_survival.csv"), help="Output CSV path")
    parser.add_argument(
        "--current-year",
        type=int,
        default=2025,
        help="Current year used for censoring duration (for ongoing shows)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_dataset(args.data_dir, args.output, args.current_year)


if __name__ == "__main__":
    main()
