.PHONY: build build-amr build-amr-openmp benchmark-smoke benchmark-openmp-threads clean test

PYTHON ?= python3

build:
	$(PYTHON) scripts/build_ext.py

build-amr:
	$(PYTHON) scripts/build_ext.py --group amr

build-amr-openmp:
	$(PYTHON) scripts/build_ext.py --group amr --openmp

test: build
	PYTHONPATH=src $(PYTHON) tests/utils/lib/test_amr.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_dataset.py
	PYTHONPATH=src $(PYTHON) tests/amrvac/test_amrvac_write.py

benchmark-smoke: build-amr
	PYTHONPATH=src $(PYTHON) -m benchmarks.amrvac_scaling --profile smoke --repetitions 1 --warmups 0 --output benchmark-results/smoke --clean-output --force-generate

benchmark-openmp-threads: build-amr-openmp
	PYTHONPATH=src $(PYTHON) -m benchmarks.amr_openmp_threads --profile smoke --threads 1,2,4 --repetitions 1 --warmups 0 --output benchmark-results/openmp-threads-smoke

clean:
	$(PYTHON) scripts/build_ext.py --clean
