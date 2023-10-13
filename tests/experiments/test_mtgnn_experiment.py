from torch_timeseries.datasets.dummy import DummyDatasetGraph
from torch_timeseries.experiments.mtgnn_experiment import MTGNNExperiment
def test_experiment():
    exp = MTGNNExperiment(
        dataset_type="ETTh1", 
        epochs=5,
        windows=384,
        horizon=3,
        pred_len=1,
        device="cuda:4"
        )
    exp.run(42)