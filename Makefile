.PHONY: build build-amr clean test

PYTHON ?= python3

build:
	$(PYTHON) scripts/build_ext.py

build-amr:
	$(PYTHON) scripts/build_ext.py --group amr

test: build
	PYTHONPATH=src $(PYTHON) tests/utils/lib/test_amr.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_dataset.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_write.py

clean:
	$(PYTHON) scripts/build_ext.py --clean
