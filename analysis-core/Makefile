PYTHON ?= .venv/bin/python

.PHONY: build test
build:
	$(PYTHON) setup.py build_ext --inplace --force

test:
	$(PYTHON) -m pytest tests -q
