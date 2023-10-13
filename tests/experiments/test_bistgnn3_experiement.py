from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.datasets import ETTh1
from torch_timeseries.experiments.bistgnnv3_experiment import BiSTGNNv3Experiment


def test_experiment():
    
    exp = BiSTGNNv3Experiment(
        dataset_type="ETTh1", 
        epochs=5,
        horizon=24,
        pred_len=1,
        windows=384,
        device="cuda:0"
        )
    exp.run(42)