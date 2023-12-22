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
                gcn_layers=wandb.config.gcn_layers,
                latent_dim=wandb.config.latent_dim,
                heads=wandb.config.heads,
                tcn_channel=wandb.config.tcn_channel,
                tn_layers=wandb.config.tn_layers,
                model_type="BiSTGNNv6",
                graph_build_type='adaptive',
                gcn_type= 'weighted_han',
                device="cuda"
    )
    result = exp.run()
    wandb.log(result)
        
    
name = "BiSTGNN_weather_sweep"

# 2: Define the search space
sweep_configuration = {
    "name": name,
    'program': '',
    'method': 'grid',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'rmse'
        },
    'parameters': 
    {
        'gcn_layers': {'values': [1,2,3,4]},
        'latent_dim': {'values': [16,32,64,128]},
        'tcn_channel': {'values': [16,32,64]},
        'self_loop_eps': {'values': [0.1,0.3,0.5,0.8,1]},
        'heads': {'values': [1,2,3,4]},
        'tn_layers': {'values': [1,2,3]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='BiSTGNN'
    )
print("sweep_id:", {sweep_id})
wandb.agent(sweep_id, function=main)

