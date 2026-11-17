PYTHON ?= python3
PYTEST_ARGS ?=

.PHONY: all simulate analyze figures clean test solver_benchmark

all: figures

setup: .venv/.deps-installed

.venv:
	$(PYTHON) -m venv .venv

.venv/.deps-installed: requirements.txt | .venv
	.venv/bin/python -m pip install --quiet --upgrade pip
	.venv/bin/python -m pip install --quiet -r requirements.txt
	touch .venv/.deps-installed

simulate: setup
	.venv/bin/python src/simulation.py

figures:
	.venv/bin/python src/figures.py

analyze:
	.venv/bin/python src/metrics.py

test: setup
	.venv/bin/python -m pytest $(PYTEST_ARGS)

profile: setup
	.venv/bin/python profiling/cprofiling.py

solver_benchmark: setup
	.venv/bin/python profiling/speed_comparison_convex_solvers.py

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
