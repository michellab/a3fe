# A3FE Integration Tests

This directory contains integration tests for A3FE, which are designed to run in a local SLURM environment rather than in a CI environment, as they may require longer run times and SLURM resources.

## Prerequisites

**For developers**: Integration tests require GROMACS and testing tools. Set up your development environment first:

```bash
cd your/path/a3fe
make env-dev  # install development environment using your local GROMACS
```

**Note**: These tests will use your locally installed GROMACS. Make sure your GROMACS is available.

## Running Integration Tests

To run integration tests, use the following command:

   ```bash
   cd your/path/a3fe
   make test-integration
   ```

Or you can run the pytest command directly:
   ```bash
   RUN_SLURM_TESTS=1 pytest a3fe/tests --run-integration -v
   ```

Or to run a specific integration test:
   ```bash
   RUN_SLURM_TESTS=1 pytest a3fe/tests/test_run_integration.py::TestSlurmIntegration::test_slurm_calculation_setup --run-integration -v
   ```

## Skipping Integration Tests

Integration tests are skipped by default. There are several ways to control which tests run:

   To skip all integration tests:
   ```bash
   make test
   ```
   
   Or using pytest directly:
   ```bash
   pytest a3fe/tests
   ```

## Test Description
The integration tests will:

1. Set up a complete SLURM calculation environment
2. Run short simulations (including adaptive and non-adaptive runs)
3. Verify output files and results
4. Test analysis functionality and job management features

These tests use example data from a3fe/data/t4l_input as input.

## Notes
1. These tests require an available SLURM environment
2. Tests will create temporary directories to store output files
3. Directories and SLURM jobs will be automatically cleaned up after tests complete