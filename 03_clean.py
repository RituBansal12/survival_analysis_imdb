#!/usr/bin/env python3
"""
Clean tvseries_survival dataset for survival analysis and produce summary stats.

Inputs:
  - data/tvseries_survival.csv (from 01_wrangle.py)

Outputs:
  - data/tvseries_survival_clean.csv
  - results/data_summary.csv

Usage:
  python 03_clean.py --input data/tvseries_survival.csv \
                     --output data/tvseries_survival_clean.csv \
                     --summary results/data_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize columns if needed
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop missing startYear
    df = df[df["startYear"].notna()]
    # Duration cleanup
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df[df["duration"].notna()]
    # Remove zero/negative
    df = df[df["duration"] > 0]

    # Handle extreme outliers: filter durations > hard cap OR far beyond IQR fence
    dur = df["duration"].astype(float)
    q1, q3 = dur.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 3.0 * iqr if pd.notna(iqr) else dur.quantile(0.99)
    hard_cap = 80.0
    max_allowed = float(min(upper_fence, hard_cap))
    df = df[dur <= max_allowed]

    # Deduplicate by tconst (keep first)
    if "tconst" in df.columns:
        df = df.sort_values(["tconst", "duration"], ascending=[True, False]).drop_duplicates("tconst", keep="first")
    else:
        df = df.drop_duplicates(subset=["title", "startYear"], keep="first")

    # Coerce event to int 0/1
    if "event" in df.columns:
        df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
        df["event"] = df["event"].clip(0, 1)

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    ended = int((df["event"] == 1).sum()) if "event" in df.columns else np.nan
    ongoing = int((df["event"] == 0).sum()) if "event" in df.columns else np.nan
    prop_censored = (ongoing / n) if (n and not np.isnan(ongoing)) else np.nan
    med_duration = float(df["duration"].median()) if "duration" in df.columns else np.nan
    med_duration_ended = float(df.loc[df["event"] == 1, "duration"].median()) if "event" in df.columns else np.nan
    med_duration_ongoing = float(df.loc[df["event"] == 0, "duration"].median()) if "event" in df.columns else np.nan

    rows = [
        {"metric": "n_rows", "value": n},
        {"metric": "n_ended", "value": ended},
        {"metric": "n_ongoing", "value": ongoing},
        {"metric": "prop_censored", "value": prop_censored},
        {"metric": "median_duration", "value": med_duration},
        {"metric": "median_duration_ended", "value": med_duration_ended},
        {"metric": "median_duration_ongoing", "value": med_duration_ongoing},
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean survival dataset and summarize")
    p.add_argument("--input", type=Path, default=Path("data/tvseries_survival.csv"))
    p.add_argument("--output", type=Path, default=Path("data/tvseries_survival_clean.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/data_summary.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load(args.input)
    dfc = basic_clean(df)
    ensure_parent(args.output)
    dfc.to_csv(args.output, index=False)

    sm = summarize(dfc)
    ensure_parent(args.summary)
    sm.to_csv(args.summary, index=False)
    print(f"Cleaned rows: {len(dfc):,}. Saved {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
