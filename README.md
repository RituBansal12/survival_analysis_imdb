# survival_analysis_imdb
Use IMDb dataset to implement survival analysis algorithms

## Data
Location: https://datasets.imdbws.com/

### IMDb datasets used
Each dataset is contained in a gzipped, tab-separated-values (TSV) formatted file in the UTF-8 character set. The first line in each file contains headers that describe what is in each column. A `\N` is used to denote that a particular field is missing or null for that title/name. The datasets of interest are as follows:

#### title.basics.tsv
- tconst (string) — alphanumeric unique identifier of the title
- titleType (string) — type/format of the title (movie, short, tvseries, tvepisode, ...)
- primaryTitle (string) — common/promotional title
- originalTitle (string) — original language title
- isAdult (boolean) — 0 non-adult; 1 adult
- startYear (YYYY) — release/start year (series start year for TV)
- endYear (YYYY) — TV series end year; `\N` otherwise
- runtimeMinutes (int) — primary runtime in minutes
- genres (string array) — up to three genres associated with the title

#### title.episode.tsv
- tconst (string) — alphanumeric identifier of episode
- parentTconst (string) — alphanumeric identifier of the parent TV Series
- seasonNumber (int) — season number the episode belongs to
- episodeNumber (int) — episode number in the TV series

#### title.ratings.tsv
- tconst (string) — alphanumeric unique identifier of the title
- averageRating (float) — weighted average of user ratings
- numVotes (int) — number of votes

## Data wrangling (01_wrangle.py)

- __Sources__
  - `title.basics.tsv`: filter `titleType == tvSeries`; keep `tconst`, `primaryTitle` (as `title`), `startYear`, `endYear`, `genres`.
  - `title.ratings.tsv`: merge on `tconst`; keep `averageRating`, `numVotes`.
  - `title.episode.tsv`: group by `parentTconst` to compute per-show aggregates.

- **Variables**
  - `tconst`: unique show ID from basics.
  - `title`: `primaryTitle` from basics.
  - `startYear`: from basics; rows with missing are dropped.
  - `endYear`: from basics; may be null.
  - `event`: 1 if `endYear` present (show ended), else 0 (ongoing/censored).
  - `duration`: if `event==1` then `endYear - startYear`; else `currentYear - startYear` (default `2025`).
  - `genres`: raw comma-separated string (up to 3 values).
  - `genre_1`, `genre_2`, `genre_3`: split of `genres` by comma; fewer than three values leave remaining columns null.
  - `averageRating`, `numVotes`: from ratings; may be null.
  - `numEpisodes`: count of episodes per show (`title.episode.tsv` grouped by `parentTconst`).
  - `maxSeason`: max `seasonNumber` per show, ignoring missing.

- **Output**
  - CSV: `data/tvseries_survival.csv` (UTF-8).
  - Re-run: `./.venv/bin/python 01_wrangle.py --data-dir data --output data/tvseries_survival.csv --current-year 2025`