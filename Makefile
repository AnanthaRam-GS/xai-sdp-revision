# Replication package: one-command entry points (see REPLICATION.md).
PYTHON ?= .venv/bin/python

.PHONY: help env reproduce quick paper-assets check-numbers test fresh-clone-check archive

help:
	@echo "make env            create .venv from requirements.lock (needs uv)"
	@echo "make reproduce      run steps 1-8 from the raw data (about 6 h on 10 cores)"
	@echo "make quick          reduced run (ant, synapse; 2 seeds) into results_quick/ and reports_quick/"
	@echo "make paper-assets   regenerate paper/figures, paper/tables and paper/numbers.tex, then check the numbers"
	@echo "make test           run the test suite"
	@echo "make archive        build dist/replication_package_v1.0.0.zip"

env:
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python --require-hashes -r requirements.lock
	uv pip install --python .venv/bin/python --no-deps -e .

reproduce:
	$(PYTHON) -m scripts.reproduce --mode full

quick:
	$(PYTHON) -m scripts.reproduce --mode quick

paper-assets:
	$(PYTHON) -m scripts.reproduce --mode paper-assets

check-numbers:
	$(PYTHON) -m scripts.check_numbers

test:
	$(PYTHON) -m pytest -q

fresh-clone-check:
	$(PYTHON) -m scripts.fresh_clone_check

archive:
	$(PYTHON) -m scripts.build_archive
