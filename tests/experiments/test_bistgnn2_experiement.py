from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.experiments.bistgnnv2_experiment import BiSTGNNv2Experiment
def test_experiment():
    
    exp = BiSTGNNv2Experiment(
        dataset_type="DummyWithTime", 
        epochs=5,
        windows=16,
        device="cuda:4"
        )
    exp.run(42)