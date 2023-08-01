

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.dlinear_experiment import DLinearExperiment


def main():
    datasets = [ "ETTm1", "ETTm2", "ETTh1", "ETTh2"] #"ExchangeRate",
    horizons = [3, 6, 12, 24]
    
    for horizon in horizons:
        exp = DLinearExperiment(
            epochs=100,
            patience=5,
            windows=90,
            horizon=horizon,
            dataset_type="ExchangeRate",
            device="cuda:4",
            )
        exp.config_wandb("BiSTGNN", "baseline")
        exp.runs()
        wandb.finish()

    
    for dataset in datasets:
        for horizon in horizons:
            exp = DLinearExperiment(
                epochs=100,
                patience=5,
                windows=384,
                horizon=horizon,
                dataset_type=dataset,
                device="cuda:4",
                )
            exp.config_wandb("BiSTGNN", "baseline")
            exp.runs()
            wandb.finish()



if __name__ == "__main__":
    main()