from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.datasets import ETTh1, DummyWithTime
from torch_timeseries.experiments.bistgnnv6_experiment import BiSTGNNv6Experiment

def test_dummy_experiment():
    
    exp = BiSTGNNv6Experiment(
        dataset_type="DummyWithTime", 
        epochs=1,
        output_layer_type='tcn6',
        horizon=24,
        pred_len=3,
        windows=12,
        device="cuda:0"
        )
    exp.run(42)
    import pdb;pdb.set_trace()
    print(exp.metrics['mape'].y_pred.shape)



def test_singlestep_experiment():
    
    exp = BiSTGNNv6Experiment(
        dataset_type="ETTh1", 
        epochs=5,
        output_layer_type='tcn6',
        horizon=24,
        pred_len=1,
        windows=384,
        device="cuda:0"
        )
    exp.run(42)




def test_multistep_experiment():
    
    exp = BiSTGNNv6Experiment(
        dataset_type="PEMS_BAY", 
        epochs=5,
        output_layer_type='tcn6',
        horizon=1,
        pred_len=3,
        windows=12,
        device="cuda:0"
        )
    exp.run(42)