# Import the W&B Python Library and log into W&B
import numpy as np
import wandb
import yaml
wandb.login()

from torch_timeseries.experiments.bistgnnv2_experiment import BiSTGNNv2Experiment


def main():
    # Set up your default hyperparameters
    with open("/notebooks/pytorch_timeseries/examples/bistgnn_sweep_parallel.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)
    exp = BiSTGNNv2Experiment(
                dataset_type="METR_LA",
                horizon=1,
                pred_len=12,
                windows=12,
                heads=wandb.config.heads,
                tcn_channel=wandb.config.tcn_channel,
                tn_layers=wandb.config.tn_layers,
                model_type="BiSTGNN",
                graph_build_type='adaptive',
                gcn_type= 'han',
    )
    
    metric_mean_std=    exp.runs(seeds=[42,233])
    
    for index, row in metric_mean_std.iterrows():
        wandb.run.summary[f"{index}_mean"] = row["mean"]
        wandb.run.summary[f"{index}_std"] = row["std"]
        wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"

        wandb.log({
            f"{index}_mean":  row["mean"],
            f"{index}_std":  row["std"],
            index: f"{row['mean']:.4f}±{row['std']:.4f}"
        })

main()