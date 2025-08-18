# Makefile for IMDb TV series survival analysis pipeline
# Usage:
#   make pipeline                  # run full pipeline 01–09
#   make visualize                 # run only distributions (02)
#   make validate BOOTSTRAP=500    # override defaults

PY ?= .venv/bin/python
DATA_RAW := data/tvseries_survival.csv
DATA_CLEAN := data/tvseries_survival_clean.csv

# Parameters
TOP_GENRES_SURV ?= 8
TOP_GENRES_COX ?= 10
TEST_SIZE ?= 0.2
SEED ?= 42
T0 ?= 5.0
BOOTSTRAP ?= 200
CURRENT_YEAR ?= 2025

.PHONY: pipeline wrangle clean_data visualize survival_eda group_tests cox parametric validate blog

pipeline:
	@echo "[pipeline] Starting full pipeline 01–09"
	$(MAKE) wrangle
	$(MAKE) visualize
	$(MAKE) clean_data
	$(MAKE) survival_eda
	$(MAKE) group_tests
	$(MAKE) cox
	$(MAKE) parametric
	$(MAKE) validate
	$(MAKE) blog
	@echo "[pipeline] Done. Results and figures are in results/ and visualizations/"

wrangle:
	$(PY) 01_wrangle.py \
		--data-dir data \
		--output $(DATA_RAW) \
		--current-year $(CURRENT_YEAR)

clean_data:
	$(PY) 03_clean.py \
		--input $(DATA_RAW) \
		--output $(DATA_CLEAN) \
		--summary results/data_summary.csv

visualize:
	$(PY) 02_visualize.py \
		--input $(DATA_RAW) \
		--outdir visualizations/distributions

survival_eda:
	$(PY) 04_survival_eda.py \
		--input $(DATA_CLEAN) \
		--vizdir visualizations/survival \
		--km-table results/km_tables.csv \
		--top-genres $(TOP_GENRES_SURV)

group_tests:
	$(PY) 05_group_tests.py \
		--input $(DATA_CLEAN) \
		--out results/logrank_tests.csv \
		--heatmap visualizations/survival/logrank_pvalues_heatmap.png \
		--top-genres $(TOP_GENRES_SURV)

cox:
	$(PY) 06_cox_model.py \
		--input $(DATA_CLEAN) \
		--results results/cox_summary.csv \
		--vizdir visualizations/cox \
		--top-genres $(TOP_GENRES_COX)

parametric:
	$(PY) 07_parametric_models.py \
		--input $(DATA_CLEAN) \
		--out results/parametric_comparison.csv \
		--viz visualizations/parametric/overlay_overall.png

validate:
	$(PY) 08_validate.py \
		--input $(DATA_CLEAN) \
		--metrics results/validation_metrics.json \
		--vizdir visualizations/validation \
		--test-size $(TEST_SIZE) \
		--seed $(SEED) \
		--top-genres $(TOP_GENRES_COX) \
		--t0 $(T0) \
		--bootstrap $(BOOTSTRAP)

blog:
	$(PY) 09_blog_plots.py \
		--input $(DATA_CLEAN) \
		--top-genres $(TOP_GENRES_SURV) \
		--cox-summary results/cox_summary.csv
