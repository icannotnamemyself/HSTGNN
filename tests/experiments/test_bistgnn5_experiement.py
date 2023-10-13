from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.datasets import ETTh1, DummyWithTime
from torch_timeseries.experiments.bistgnnv5_experiment import BiSTGNNv5Experiment

def test_dummy_experiment():
    
    exp = BiSTGNNv5Experiment(
        dataset_type="DummyWithTime", 
        epochs=5,
        output_layer_type='tcn3',
        horizon=24,
        pred_len=1,
        windows=12,
        device="cuda:0"
        )
    exp.run(42)



def test_singlestep_experiment():
    
    exp = BiSTGNNv5Experiment(
        dataset_type="ETTh1", 
        epochs=5,
        output_layer_type='tcn3',
        horizon=24,
        pred_len=1,
        windows=384,
        device="cuda:0"
        )
    exp.run(42)




def test_multistep_experiment():
    exp = BiSTGNNv5Experiment(
        dataset_type="PEMS_BAY", 
        graph_build_type='predefined_adaptive',
        epochs=5,
        output_layer_type='tcn4',
        horizon=1,
        pred_len=3,
        windows=12,
        device="cuda:0"
        )
    exp.run(42)