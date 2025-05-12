.PHONY: build utils amr clean test test-amr

# Define dependencies
amr: utils

# Basic build commands
utils:
	python scripts/build_ext.py --group utils

amr:
	python scripts/build_ext.py --group amr

# Build all
build: utils amr

# Testing targets
test-amr: amr
	PYTHONPATH=. python tests/utils/lib/test_amr.py

test: test-amr
	# Add other test targets as needed

# Utilities
clean:
	python scripts/build_ext.py --clean