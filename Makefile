PYTHON ?= python3
SIM_ARGS ?=
ANALYZE_ARGS ?=

.PHONY: all simulate analyze figures clean test

all: figures

setup: .venv

.venv:
	$(PYTHON) -m venv .venv

simulate: setup
	.venv/bin/python src/simulation.py

# analyze: simulate
# 	.venv/bin/python src.metrics

# figures: analyze
# 	.venv/bin/python src/metrics

test: setup
	.venv/bin/python tests/test_stats_methods.py

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
