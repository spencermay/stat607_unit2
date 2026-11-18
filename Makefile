PYTHON ?= python3
PYTEST_ARGS ?=

.PHONY: all simulate analyze figures timing_comparison clean test solver_benchmark create_profile profiles stop_profiles

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

timing_comparison:
	.venv/bin/python src/figures.py timing_comparison

analyze:
	.venv/bin/python src/metrics.py

test: setup
	.venv/bin/python -m pytest $(PYTEST_ARGS)

create_profile: setup
	.venv/bin/python profiling/cprofiling.py

# The "before" profile is a fixed historical snapshot (one recorded file), so
# its path is a literal. The "after" profile is named with the current short
# git hash by cprofiling.py
# Used Claude's help with the hash-dependent behavior
GIT_SHORT_HASH := $(shell git rev-parse --short HEAD)
BEFORE_PROF ?= profiling/profiles/before_dependent_calculationa49d7cb.prof
AFTER_PROF  ?= profiling/profiles/after_speedup_dependent_calculation$(GIT_SHORT_HASH).prof

# Regenerate the after-profile only when the file for the current hash is
# missing; cprofiling.py writes the hash-suffixed name itself.
$(AFTER_PROF): | setup
	.venv/bin/python profiling/cprofiling.py

# Run in the background both before/after snakeviz servers (each opens a browser tab),
# print their URLs + PIDs, and let the target return. Stop them with `make stop_profiles`.
profiles: $(AFTER_PROF)
	@.venv/bin/snakeviz -p 8080 $(BEFORE_PROF) & echo "before -> http://127.0.0.1:8080 (pid $$!)"
	@.venv/bin/snakeviz -p 8081 $(AFTER_PROF)  & echo "after  -> http://127.0.0.1:8081 (pid $$!)"
	@echo "Two snakeviz servers running in the background. Stop with 'make stop_profiles'."

stop_profiles:
	@pkill -f "snakeviz -p 808" && echo "stopped snakeviz servers" || echo "no snakeviz servers running"

solver_benchmark: setup
	.venv/bin/python profiling/speed_comparison_convex_solvers.py

clean:
	rm -rf results/raw results/figures
	rm -f results/summary.csv results/paired.csv results/evaluation.json results/metrics_source.py
	rm -rf profiling/profiles/after_speedup*