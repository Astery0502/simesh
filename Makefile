.PHONY: build build-amr benchmark-smoke clean test

PYTHON ?= python3

build:
	$(PYTHON) scripts/build_ext.py

build-amr:
	$(PYTHON) scripts/build_ext.py --group amr

test: build
	PYTHONPATH=src $(PYTHON) tests/utils/lib/test_amr.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_dataset.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_write.py

benchmark-smoke: build-amr
	PYTHONPATH=src $(PYTHON) -m benchmarks.amrvac_scaling --profile smoke --repetitions 1 --warmups 0 --output benchmark-results/smoke --clean-output --force-generate

clean:
	$(PYTHON) scripts/build_ext.py --clean
