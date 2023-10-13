from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.datasets import ETTh1
from torch_timeseries.experiments.bistgnnv4_experiment import BiSTGNNv4Experiment


def test_singlestep_experiment():
    
    exp = BiSTGNNv4Experiment(
        dataset_type="ETTh1", 
        epochs=5,
        output_layer_type='tcn4',
        horizon=24,
        pred_len=1,
        windows=384,
        device="cuda:0"
        )
    exp.run(42)




def test_multistep_experiment():
    
    exp = BiSTGNNv4Experiment(
        dataset_type="PEMS_BAY", 
        epochs=5,
        output_layer_type='tcn4',
        horizon=1,
        pred_len=3,
        windows=12,
        device="cuda:0"
        )
    exp.run(42)