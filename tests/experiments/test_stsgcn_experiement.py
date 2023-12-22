from torch_timeseries.datasets.dummy import DummyWithTime
from torch_timeseries.datasets import ETTh1, DummyWithTime
from torch_timeseries.experiments.STSGCN_experiment import STSGCNExperiment

def test_multistep_experiment():
    
    exp = STSGCNExperiment(
        dataset_type="PEMS04", 
        epochs=1,
        horizon=1,
        pred_len=3,
        windows=12,
        device="cuda:3"
        )
    exp.run(42)


