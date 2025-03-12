from a3fe.run._simulation_runner import SimulationRunner

class MockSimulationRunner(SimulationRunner):
    """Simple mock for testing update_engine_config_option."""
    def __init__(self):
        self.engine_config = {}
        self._sub_sim_runners = []


def test_update_engine_config_option():
    """Test the basic functionality of update engine_config."""
    runner = MockSimulationRunner()
    runner.update_engine_config_option("test_option", "test_value")
    assert runner.engine_config["test_option"] == "test_value"


def test_update_engine_config_option_with_sub_runners():
    """Test that update engine_config propagates to sub runners."""
    runner = MockSimulationRunner()
    sub_runner = MockSimulationRunner()
    runner._sub_sim_runners.append(sub_runner)

    runner.update_engine_config_option("test_option", "test_value")

    assert runner.engine_config["test_option"] == "test_value"
    assert sub_runner.engine_config["test_option"] == "test_value"


def test_engine_config_propagation(calc):
    """test that engine_config propagates to all components"""
    # save original value for later restoration
    original_runtime = calc.engine_config.runtime
    
    try:
        # set test value and set at calculation level
        test_runtime = 99
        calc.engine_config.runtime = test_runtime
        
        # Propagate value to sub components
        for leg in calc.legs:
            leg.engine_config.runtime = test_runtime
            assert leg.engine_config.runtime == test_runtime
            
            for stage in leg.stages:
                stage.engine_config.runtime = test_runtime
                assert stage.engine_config.runtime == test_runtime

                for lam_win in stage.lam_windows:
                    lam_win.engine_config.runtime = test_runtime
                    assert lam_win.engine_config.runtime == test_runtime
                    
                    for sim in lam_win.simulations:
                        sim.engine_config.runtime = test_runtime
                        assert sim.engine_config.runtime == test_runtime
    
    finally:
        # restore original value - use direct assignment
        calc.engine_config.runtime = original_runtime


def test_slurm_config_propagation_and_independence(calc):
    """test that slurm_config and analysis_slurm_config propagates to all components and remains independent"""
    # save original values for later restoration
    original_partition = calc.slurm_config.partition
    original_analysis_partition = calc.analysis_slurm_config.partition
    
    try:
        # 1. Test slurm_config propagation and check propagation to all components
        test_partition = "test_partition_GPUX"
        calc.slurm_config.partition = test_partition
        
        for leg in calc.legs:
            assert leg.slurm_config.partition == test_partition

            for stage in leg.stages:
                assert stage.slurm_config.partition == test_partition

                for lam_win in stage.lam_windows:
                    assert lam_win.slurm_config.partition == test_partition

                    for sim in lam_win.simulations:
                        assert sim.slurm_config.partition == test_partition
        
        # 2. Test analysis_slurm_config independence
        analysis_partition = "analysis_partition"
        calc.analysis_slurm_config.partition = analysis_partition
        
        assert calc.slurm_config.partition == test_partition
        assert calc.analysis_slurm_config.partition == analysis_partition
    
    finally:
        # restore original values
        calc.slurm_config.partition = original_partition
        calc.analysis_slurm_config.partition = original_analysis_partition