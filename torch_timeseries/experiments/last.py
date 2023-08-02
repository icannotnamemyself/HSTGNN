from typing import Dict, List, Type
import fire
import numpy as np
import torch
from tqdm import tqdm
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import LaST
import wandb

from dataclasses import dataclass, asdict


@dataclass
class LaSTExperiment(Experiment):
    model_type: str = "LaST"
    
    var_num : int = 1
    latent_dim : int = 64
    para_mode  : int = 0
    dropout : float = 0.0
    
    def _init_model(self):
        self.model = LaST(
                input_len=self.windows,
                output_len=self.pred_len,
                input_dim=self.dataset.num_features,
                out_dim=self.dataset.num_features,
                var_num=self.var_num,
                latent_dim=self.latent_dim,
                dropout=self.dropout,
                device=self.device
            )
        self.model = self.model.to(self.device)


    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        outputs, _, _, _ = self.model(batch_x)
        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        return outputs, batch_y

    def _process_one_batch_train(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        outputs, elbo, mlbo, mubo = self.model(batch_x)
        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        return outputs, batch_y, elbo, mlbo, mubo



    def _train(self):
        with torch.enable_grad(), tqdm(total=self.train_steps) as progress_bar:
            self.model.train()
            for i, (
                batch_x,
                batch_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                for para_mode in range(2):
                    self.model_optim.zero_grad()
                    if para_mode == 0:
                        for para in self.model.parameters():
                            para.requires_grad = True

                        for para in self.model.LaSTLayer.MuboNet.parameters():
                            para.requires_grad = False
                        for para in self.model.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                            para.requires_grad = False
                        for para in self.model.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                            para.requires_grad = False

                    elif para_mode == 1:
                        for para in self.model.parameters():
                            para.requires_grad = False

                        for para in self.model.LaSTLayer.MuboNet.parameters():
                            para.requires_grad = True
                        for para in self.model.LaSTLayer.SNet.VarUnit_s.critic_xz.parameters():
                            para.requires_grad = True
                        for para in self.model.LaSTLayer.TNet.VarUnit_t.critic_xz.parameters():
                            para.requires_grad = True 
                    pred, true, elbo, mlbo, mubo = self._process_one_batch_train( batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)
                    
                    if self.invtrans_loss:
                        pred = self.scaler.inverse_transform(pred)
                        batch_y = self.scaler.inverse_transform(batch_y)
                    else:
                        pred = pred
                        batch_y = batch_y
                    loss = (self.loss_func(pred, true) - elbo - mlbo + mubo) if self.para_mode == 0 else (mubo - mlbo)

                    
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )


                    if self._use_wandb():
                        wandb.log({"loss": loss.item()}, step=i)

                    self.model_optim.step()
                
                
                progress_bar.update(batch_x.size(0))
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )


def main():
    exp = LaSTExperiment(epochs=3,dataset_type="ExchangeRate", windows=96)
    exp.run()


if __name__ == "__main__":
    # fire.Fire(LaSTExperiment)
    main()
    
    
