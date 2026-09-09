PYTHON ?= .venv/bin/python

.PHONY: build test docs docs-check docs-serve
build:
	$(PYTHON) setup.py build_ext --inplace --force

test:
	$(PYTHON) -m pytest tests -q

docs-check:
	$(PYTHON) scripts/check_docs.py

docs: docs-check
	$(PYTHON) -m mkdocs build --strict

docs-serve: docs-check
	$(PYTHON) -m mkdocs serve
