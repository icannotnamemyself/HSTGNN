

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.film_experiment import FiLMExperiment


def main():
    datasets = [ "ETTm2", "ETTh1", "ETTh2"] #"ExchangeRate","ETTm1", 
    horizons = [3, 6, 12, 24]
    device = "cuda:3"
    
    for horizon in horizons:
        exp = FiLMExperiment(
            epochs=100,
            patience=5,
            windows=90,
            horizon=horizon,
            dataset_type="ExchangeRate",
            device=device,
            )
        exp.config_wandb("BiSTGNN", "baseline")
        exp.runs()
        wandb.finish()

    
    for dataset in datasets:
        for horizon in horizons:
            exp = FiLMExperiment(
                epochs=100,
                patience=5,
                windows=384,
                horizon=horizon,
                dataset_type=dataset,
                device=device,
                )
            exp.config_wandb("BiSTGNN", "baseline")
            exp.runs()
            wandb.finish()



if __name__ == "__main__":
    main()