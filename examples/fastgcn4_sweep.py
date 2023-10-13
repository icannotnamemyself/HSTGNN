# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.tntcn_experiment import TNTCNExperiment

def main():
    wandb.init(project='BiSTGNN')
    config = wandb.config
    exp = TNTCNExperiment(
                dataset_type="ETTm1",
                horizon=3,
                windows=384,
                model_type="fastgcn4_sweep",
                casting_dim=config.casting_dim,
                gcn_channel=config.gcn_channel,
                gc_layers=config.gc_layers,
                graph_build_type='nt_full_connected',
                gcn_type= 'heterofastgcn4',
                device="cuda:6",
                                   )
    metric_mean_std = exp.runs()
    
    for index, row in metric_mean_std.iterrows():
        wandb.run.summary[f"{index}_mean"] = row["mean"]
        wandb.run.summary[f"{index}_std"] = row["std"]
        wandb.run.summary[index] = f"{row['mean']:.4f}±{row['std']:.4f}"

        wandb.log({
            f"{index}_mean":  row["mean"],
            f"{index}_std":  row["std"],
            index: f"{row['mean']:.4f}±{row['std']:.4f}"
        })
        
        
    
name = "fastgcn4_sweep"

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
        'gcn_channel': {'values': [32,128,512]},
        'casting_dim': {'values': [32,128,512]},
        'gc_layers': {'values': [2,4,6]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='BiSTGNN'
    )

wandb.agent(sweep_id, function=main, count=50)

