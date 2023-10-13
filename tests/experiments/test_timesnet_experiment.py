from torch_timeseries.experiments.timesnet_experiment import TimesNetExperiment


def test_experiment():
    
    exp = TimesNetExperiment(
        d_ff=16,
        d_model=16,
        e_layers=1,
        dataset_type="ExchangeRate", 
        epochs=1
        )
    exp._setup_run(42)
    exp._test()