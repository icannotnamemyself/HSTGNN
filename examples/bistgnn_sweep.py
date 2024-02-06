# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.bistgnnv2_experiment import BiSTGNNv2Experiment

def main():
    wandb.init(project='BiSTGNN')
    config = wandb.config
    exp = BiSTGNNv2Experiment(
                dataset_type="METR_LA",
                horizon=1,
                pred_len=12,
                windows=12,
                gcn_layers=config.gcn_layers,
                model_type="BiSTGNN",
                graph_build_type='adaptive',
                gcn_type= 'han',
                device="cuda:2",
                                   )
    metric_mean_std = exp.runs(seeds=[42,233])
    
    for index, row in metric_mean_std.iterrows():
        wandb.run.summary[f"{index}_mean"] = row["mean"]
        wandb.run.summary[f"{index}_std"] = row["std"]
        wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"

        wandb.log({
            f"{index}_mean":  row["mean"],
            f"{index}_std":  row["std"],
            index: f"{row['mean']:.4f}±{row['std']:.4f}"
        })
        
        
    
name = "BiSTGNN_sweep"

# 2: Define the search space
sweep_configuration = {
    "name": name,
    'method': 'grid',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'gcn_layers': {'values': [1,2,3]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='BiSTGNN'
    )

wandb.agent(sweep_id, function=main, count=50)

