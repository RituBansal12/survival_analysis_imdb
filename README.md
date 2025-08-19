# IMDb TV Series Survival Analysis

Analyze the “lifetimes” of TV series using the public IMDb datasets with classical survival analysis methods. This repository provides an end‑to‑end, reproducible pipeline: data wrangling, exploratory survival analysis, hypothesis tests, Cox proportional hazards modeling, parametric models, validation, and publication‑ready figures.

---

## Table of Contents

1. [Overview](#overview)
2. [Articles / Publications](#articles--publications)
3. [Project Workflow](#project-workflow)
4. [File Structure](#file-structure)
5. [Data Directory](#data-directory)
6. [Visualizations / Outputs](#visualizations--outputs)
7. [Key Concepts / Variables](#key-concepts--variables)
8. [Installation and Setup](#installation-and-setup)
9. [Usage](#usage)
10. [Results / Interpretation](#results--interpretation)
11. [Technical Details](#technical-details)
12. [Dependencies](#dependencies)
13. [Notes / Limitations](#notes--limitations)

---

## Overview

* **Goal**: Quantify how long TV series run, what factors are associated with ending sooner vs later, and how survival differs across genres and other attributes.
* **Approach**: Kaplan–Meier estimates, logrank tests, Cox PH regression, and parametric survival models, implemented with `lifelines` and `scikit-learn` utilities.
* **Highlights**:
  - One‑command pipeline via `Makefile` (steps 01–09)
  - Reproducible figures and result tables saved under `visualizations/` and `results/`
  - Configurable parameters (e.g., top genres, validation settings, censoring year)

---

## Articles / Publications

* (TBD) Add links to blog post

---

## Project Workflow

1. **Data Collection / Extraction**: Download IMDb TSV files and derive a per‑show dataset with duration and event indicators.
2. **Data Preprocessing / Cleaning**: Split multi‑genre strings, derive groups, remove obvious inconsistencies, summarize data.
3. **Modeling / Analysis**: KM curves, group comparisons, Cox PH model, and parametric fits with overlays.
4. **Evaluation / Validation**: Train/validation split, concordance, calibration at horizon `T0`, and optional bootstrap.
5. **Visualization / Reporting**: Static figures for distributions, survival, Cox, parametric overlays, and validation.

---

## File Structure

### Core Scripts

#### `01_wrangle.py`
* Purpose: Build per‑series dataset from IMDb TSVs; compute `duration` and `event`.
* Input: `data/title.basics.tsv`, `data/title.ratings.tsv`, `data/title.episode.tsv`
* Output: `data/tvseries_survival.csv`

#### `02_visualize.py`
* Purpose: Descriptive distribution plots from the raw assembled dataset.
* Input: `data/tvseries_survival.csv`
* Output: Figures in `visualizations/distributions/`

#### `03_clean.py`
* Purpose: Clean features, engineer groups, and summarize.
* Input: `data/tvseries_survival.csv`
* Output: `data/tvseries_survival_clean.csv`, `results/data_summary.csv`

#### `04_survival_eda.py`
* Purpose: KM survival curves overall and by groups; cumulative hazard.
* Input: `data/tvseries_survival_clean.csv`
* Output: KM tables `results/km_tables.csv`, figures in `visualizations/survival/`

#### `05_group_tests.py`
* Purpose: Logrank tests across top genres with heatmap.
* Input: `data/tvseries_survival_clean.csv`
* Output: `results/logrank_tests.csv`, heatmap in `visualizations/survival/`

#### `06_cox_model.py`
* Purpose: Cox proportional hazards model and diagnostics.
* Input: `data/tvseries_survival_clean.csv`
* Output: `results/cox_summary.csv`, figures in `visualizations/cox/`

#### `07_parametric_models.py`
* Purpose: Compare parametric survival families and overlay with KM.
* Input: `data/tvseries_survival_clean.csv`
* Output: `results/parametric_comparison.csv`, overlay in `visualizations/parametric/`

#### `08_validate.py`
* Purpose: Holdout validation, concordance, calibration, optional bootstrap.
* Input: `data/tvseries_survival_clean.csv`
* Output: `results/validation_metrics.json`, figures in `visualizations/validation/`

#### `09_blog_plots.py`
* Purpose: Publication‑style versions of key figures (forest, hazard shape, KM by genre).
* Input: `data/tvseries_survival_clean.csv`, `results/cox_summary.csv`
* Output: Figures saved under `visualizations/` subfolders

---

## Data Directory

IMDb datasets: https://datasets.imdbws.com/

Place the following raw files under `data/` (gzip TSVs from IMDb):
* `title.basics.tsv` — includes `titleType`, `primaryTitle`, `startYear`, `endYear`, `genres`
* `title.ratings.tsv` — includes `averageRating`, `numVotes`
* `title.episode.tsv` — includes `parentTconst`, `seasonNumber`, `episodeNumber`

Outputs and intermediate CSVs are also stored under `data/`.

---

## Visualizations / Outputs

* `visualizations/distributions/` — histograms, correlations, group summaries
* `visualizations/survival/` — KM survival, cumulative hazard, group curves
* `visualizations/cox/` — forest plot, PH diagnostics
* `visualizations/parametric/` — KM overlay for parametric fits
* `visualizations/validation/` — calibration at `T0`, bootstrap histograms (optional)

Result tables in `results/`:
* `data_summary.csv`, `km_tables.csv`, `logrank_tests.csv`, `cox_summary.csv`, `parametric_comparison.csv`, `validation_metrics.json`

---

## Key Concepts / Variables

* **event**: 1 if a series ended (`endYear` present), else 0 (censored/ongoing).
* **duration**: If `event==1`, `endYear - startYear`; otherwise `currentYear - startYear`.
* **groups**: Top genres, start decade, and rating bands for stratified analysis.
* **episodes/season features**: `numEpisodes`, `maxSeason` aggregated from `title.episode.tsv`.

---

## Installation and Setup

1. Clone and create a virtual environment

   ```bash
   git clone <repo-url>
   cd survival_analysis_imdb
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare data
   * Download IMDb TSVs from https://datasets.imdbws.com/ and place them in `data/`
   * Ensure filenames match exactly (e.g., `title.basics.tsv`)

---

## Usage

### Run Complete Pipeline (recommended)

```bash
# Uses defaults from Makefile: TOP_GENRES_SURV=8, TOP_GENRES_COX=10,
# TEST_SIZE=0.2, SEED=42, T0=5.0, BOOTSTRAP=200, CURRENT_YEAR=2025
make pipeline
```

Override parameters at invocation, e.g.:

```bash
make validate BOOTSTRAP=500 T0=7.5
make pipeline CURRENT_YEAR=2024
```

### Run Individual Steps

```bash
make wrangle
make visualize
make clean_data
make survival_eda
make group_tests
make cox
make parametric
make validate
make blog
```

Each target maps to a script with CLI flags (see `Makefile`). For example:

```bash
./.venv/bin/python 04_survival_eda.py \
  --input data/tvseries_survival_clean.csv \
  --vizdir visualizations/survival \
  --km-table results/km_tables.csv \
  --top-genres 8
```

---

## Results / Interpretation

* Expect populated CSV/JSON artifacts in `results/` and figures across `visualizations/` subfolders.
* Validation reports concordance and calibration at `T0` (see `results/validation_metrics.json`).
* Use `09_blog_plots.py` outputs for publication or blogging.

---

## Technical Details

* **Algorithms / Models**: Kaplan–Meier, logrank tests, Cox PH, and several standard parametric survival families.
* **Frameworks / Tools**: `lifelines`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
* **Implementation Notes**: One‑command orchestration via `Makefile`; seeds and key parameters are configurable as environment overrides.

---

## Dependencies

See `requirements.txt` for exact versions. Key libraries:
* `pandas`, `numpy`, `pyarrow`
* `lifelines`
* `scikit-learn`, `scipy`
* `matplotlib`, `seaborn`

---

## Notes / Limitations

* IMDb data are observational; many covariates relevant to show longevity are unobserved.
* Multi‑genre labeling is simplified (top‑k genres); results may vary with different encodings.
* Ongoing shows are right‑censored at `CURRENT_YEAR`; adjust as needed via `Makefile`.
* Cox PH assumes proportional hazards; inspect diagnostics and interpret cautiously.