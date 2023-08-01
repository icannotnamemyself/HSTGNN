

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.gru_experiment import GRUExperiment


def main():
    datasets = [ "ETTm1", "ETTm2", "ETTh1", "ETTh2"] #"ExchangeRate",
    horizons = [3, 6, 12, 24]
    for dataset in datasets:
        for horizon in horizons:
            exp = GRUExperiment(
                epochs=100,
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