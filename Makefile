PACKAGE_NAME := a3fe
PACKAGE_DIR  := a3fe

# Only need slurm, which is linux only
CONDA_ENV_RUN = conda run --no-capture-output --name $(PACKAGE_NAME)

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

.PHONY: env env-dev env-ci lint format test test-integration docs-build docs clean

# Regular users:
env:
	mamba create -y --name $(PACKAGE_NAME) $(if $(PYTHON_VERSION),python=$(PYTHON_VERSION))
	mamba env update --name $(PACKAGE_NAME) --file devtools/conda-envs/base_env.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .

# Developers with local GROMACS
env-dev:
	mamba create -y --name $(PACKAGE_NAME) $(if $(PYTHON_VERSION),python=$(PYTHON_VERSION))
	mamba env update --name $(PACKAGE_NAME) --file devtools/conda-envs/dev_env.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

# CI and developers without GROMACS
env-ci:
	mamba create -y --name $(PACKAGE_NAME) $(if $(PYTHON_VERSION),python=$(PYTHON_VERSION))
	mamba env update --name $(PACKAGE_NAME) --file devtools/conda-envs/ci_env.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true
	
# code check
lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_DIR)

# code format
format:
	$(CONDA_ENV_RUN) ruff format $(PACKAGE_DIR)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_DIR)

# run tests
test:
	$(CONDA_ENV_RUN) pytest $(TEST_ARGS) $(PACKAGE_DIR)/tests/

# run integration tests (requires SLURM environment)
test-integration:
	$(CONDA_ENV_RUN) RUN_SLURM_TESTS=1 pytest $(TEST_ARGS) $(PACKAGE_DIR)/tests/ --run-integration -v

# build docs
docs-build:
	cd docs && $(CONDA_ENV_RUN) make html

docs:
	cd docs && $(CONDA_ENV_RUN) make html
	@echo "Documentation built in docs/_build/html/"

# clean build files
clean:
	cd docs && make clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# deploy docs
# consider supporting this in the future