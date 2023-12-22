# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

from torch_timeseries.experiments.bistgnnv6_experiment import BiSTGNNv6Experiment

def main():
    wandb.init(project='BiSTGNN')
    exp = BiSTGNNv6Experiment(
                dataset_type="Weather",
                horizon=3,
                pred_len=1,
                windows=168,
                batch_size=64,
                self_loop_eps=0.0,
                latent_dim=16,
                tcn_layers=wandb.config.tcn_layers,
                heads=wandb.config.heads,
                gcn_layers=wandb.config.gcn_layers,
                tn_layers=wandb.config.tn_layers,
                model_type="BiSTGNNv6",
                graph_build_type='adaptive',
                gcn_type= 'weighted_han',
                device="cuda:1"
    )
    result = exp.run()
    wandb.log(result)
        
    
name = "BiSTGNN_weather_sweep"

# 2: Define the search space
sweep_configuration = {
    "name": name,
    'method': 'grid',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'rmse'
        },
    'parameters': 
    {
        'gcn_layers': {'values': [1,2]},
        'tn_layers': {'values': [1,2]},
        'heads': {'values': [1,2]},
        'tcn_layers': {'values': [1,2,3]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='BiSTGNN'
    )
print("sweep_id:", {sweep_id})
wandb.agent(sweep_id, function=main)

