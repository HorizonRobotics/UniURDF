# Project uniurdf
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

PYTHON ?= python3

.PHONY: help check lint format types test wheel viewer vendor clean

help:
	@echo "make check    lint + types + test + wheel  (run this before a commit)"
	@echo "make lint     ruff check and format --check"
	@echo "make format   ruff format, in place"
	@echo "make types    pyright over src"
	@echo "make test     pytest"
	@echo "make wheel    build a wheel and check the installed viewer serves"
	@echo "make vendor   download the viewer's third-party JavaScript"
	@echo "make clean    remove build and cache directories"
	@echo ""
	@echo "PYTHON=$(PYTHON)"

check: lint types test wheel

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

types:
	$(PYTHON) -m pyright --pythonpath $(PYTHON) src

test:
	$(PYTHON) -m pytest -q

vendor:
	$(PYTHON) scripts/fetch_vendor.py

# What `pytest` alone cannot see. Package data is located relative to its
# module at run time, so a file present in the checkout can still be absent
# from the wheel -- and the tests, which read the checkout, pass either way.
# This builds the wheel, unpacks it somewhere unrelated, and asks the
# installed package to serve every asset over its real route.
wheel: vendor
	@$(PYTHON) -c "import hatchling" 2>/dev/null || { \
	    echo "make wheel needs hatchling: pip install --group dev" >&2; \
	    exit 1; }
	@rm -rf build/wheel && mkdir -p build/wheel
	$(PYTHON) -c "from hatchling.builders.wheel import WheelBuilder; \
	    print(next(iter(WheelBuilder('.').build(directory='build/wheel', versions=['standard']))))"
	@$(PYTHON) scripts/check_wheel.py

viewer:
	$(PYTHON) -m uniurdf.cli view src/uniurdf/example/example_arm.yaml \
	    --output build/viewer

clean:
	rm -rf build dist .pytest_cache .ruff_cache htmlcov coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
