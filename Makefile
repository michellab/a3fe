PACKAGE_NAME := a3fe
PACKAGE_DIR  := a3fe

# Set CONDA_ENV_RUN to empty if SKIP_CONDA_ENV_RUN is not set and GITHUB_ACTIONS is set
# For the CI github actions workflow, we skip "make env" and set up the environment manually. In this case,
# it's helpful to to set CONDA_ENV_RUN to be empty. However, for the documentation workflow, we want to override
# this and keep the normal behavior. We override this by setting KEEP_CONDA_ENV_RUN to true in the documentation workflow.
SKIP_CONDA_ENV = $(and $(GITHUB_ACTIONS),$(if $(KEEP_CONDA_ENV_RUN),,true))
CONDA_ENV_RUN = $(if $(SKIP_CONDA_ENV),,conda run --no-capture-output --name $(PACKAGE_NAME))

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

.PHONY: env lint format test type-check docs-build docs clean

# create and configure development environment
env:
	# Use the project's script to create the environment
	python devtools/scripts/create_conda_env.py -n $(PACKAGE_NAME) -p 3.9 devtools/conda-envs/test_env.yaml
	# Install development tools
	$(CONDA_ENV_RUN) pip install -e .
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

# type check
type-check:
	$(CONDA_ENV_RUN) mypy --follow-imports=silent --ignore-missing-imports --strict $(PACKAGE_DIR)

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