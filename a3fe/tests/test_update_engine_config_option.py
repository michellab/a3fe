from a3fe.run._simulation_runner import SimulationRunner


class MockSimulationRunner(SimulationRunner):
    def __init__(self):
        self.engine_config = {}
        self._sub_sim_runners = []


def test_update_engine_config_option():
    runner = MockSimulationRunner()
    runner.update_engine_config_option("test_option", "test_value")
    assert runner.engine_config["test_option"] == "test_value"


def test_update_engine_config_option_with_sub_runners():
    runner = MockSimulationRunner()
    sub_runner = MockSimulationRunner()
    runner._sub_sim_runners.append(sub_runner)

    runner.update_engine_config_option("test_option", "test_value")

    assert runner.engine_config["test_option"] == "test_value"
    assert sub_runner.engine_config["test_option"] == "test_value"
