

import random
import time
from typing import Dict, List, Type
import numpy as np
import torch
from tqdm import tqdm
from torch_timeseries.data.scaler import *
from torch_timeseries.datasets import *
from torch_timeseries.datasets.dataset import TimeSeriesDataset, TimeSeriesStaticGraphDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.experiments.experiment import Experiment
from torch_timeseries.models import STAGNN_stamp as TPGNN
from torch_timeseries.nn.metric import TrendAcc, R2, Corr
from torch.nn import MSELoss, L1Loss
from torchmetrics import MetricCollection, R2Score, MeanSquaredError
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset

import argparse
import numpy as np
import os
import pandas as pd

import os

from torch.optim import Optimizer, Adam

import wandb

from dataclasses import dataclass, asdict, field




class DefaultConfig(object):
    seed = 666
    device = 0

    day_slot = 288
    n_route, n_his, n_pred = 228, 12, 12
    # n_train, n_val, n_test = 34, 5, 5

    mode = 1
    # 1: 3, 6, 9, 12
    # 2: 3, 6, 12, 18, 24
    n_c = 10
    model = 'STAGNN_stamp'
    TPG = 'TPGNN'  
    # name = r'TPGNN_r05p02kt3outer'
    # if not os.path.exists("log"):
    #     os.makedirs("log")
    # log_path = os.path.join(
    #     "log", name)
    # if os.path.exists(log_path):
    #     crash = 1
    #     new_name = "_".join([name, str(crash)])
    #     log_path = os.path.join(
    #         "log", new_name)
    #     while os.path.exists(log_path):
    #         crash += 1
    #         new_name = "_".join([name, str(crash)])
    #         log_path = os.path.join(
    #             "log", new_name)
    #     name = new_name
    # batch_size = 50
    # lr = 1e-3

    a = 0.1
    r = 0.5
    n_mask = 1

    #optimizer
    adam = {'use': True, 'weight_decay': 1e-4}
    slr = {'use': True, 'step_size': 400, 'gamma': 0.3}

    # resume = False
    # start_epoch = 0
    # epochs = 1500

    n_layer = 1
    # n_attr, n_hid = 64, 512
    n_attr, n_hid = 64, 512
    reg_A = 1e-4
    circle = 12*24
    drop_prob = 0.2

    # expand attr by conv
    CE = {'use': True, 'kernel_size': 1, 'bias': False}
    # expand attr by linear
    LE = {'use': False, 'bias': False}
    # spatio encoding
    SE = {'use': True, 'separate': True, 'no': False}
    # tempo encoding
    TE = {'use': True, 'no': True}

    # MultiHeadAttention
    attn = {'head': 1, 'd_k': 32, 'd_v': 32, 'drop_prob': drop_prob}

    # TPGNN polynomial order
    STstamp = {'use': True, 'kt': 3, 'temperature': 1.0}

    # TeaforN
    T4N = {'use': True, 'step': 2, 'end_epoch': 10000,
           'change_head': True, 'change_enc': True}


    # # TeaforN
    # T4N = {'use': True, 'step': 2, 'end_epoch': 10000,
    #        'change_head': True, 'change_enc': True}
    # stamp_path = "data/PeMS/time_stamp.npy"
    # data_path = 'data/PeMS/V_228.csv'
    # adj_matrix_path = 'data/PeMS/W_228.csv'
    dis_mat = None

    # prefix = 'log/' + name + '/'
    # checkpoint_temp_path = prefix + '/temp.pth'
    # checkpoint_best_path = prefix + '/best.pth'
    # tensorboard_path = prefix
    # record_path = prefix + 'record.txt'

    # if not os.path.exists(prefix):
    #     os.makedirs(prefix)

    eps = 0.1

    # def parse(self, kwargs):
    #     '''
    #     customize configuration by input in terminal
    #     '''
    #     for k, v in kwargs.items():
    #         if not hasattr(self, k):
    #             warnings.warn('Warning: opt has no attribute %s' % k)
    #         setattr(self, k, v)

    # def output(self):
    #     print('user config:')
    #     for k, v in self.__class__.__dict__.items():
    #         if not k.startswith('__'):
    #             print(k, getattr(self, k))




@dataclass
class TPGNNExperiment(Experiment):
    model_type: str = "TPGNN"

    
    T4N_steps: int = 3

    def _init_data_loader(self):
        self.dataset : TimeSeriesDataset = self._parse_type(self.dataset_type)(root=self.data_path)
        self.scaler = self._parse_type(self.scaler_type)()
        self.dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            scale_in_train=False,
            shuffle_train=True,
            freq="h",
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.2,
            num_worker=self.num_worker,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = self.dataloader.train_size
        self.val_steps = self.dataloader.val_size
        self.test_steps = self.dataloader.test_size

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")


        
        
        self.train_dataloader = ChunkSequenceTimefeatureDataLoader(
            self.dataset,
            self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len+self.T4N_steps,
            scale_in_train=False,
            shuffle_train=True,
            freq="h",
            batch_size=self.batch_size,
            train_ratio=0.7,
            val_ratio=0.2,
            num_worker=self.num_worker,
        )
        self.train_steps = self.train_dataloader.train_size
        print(f"train steps: {self.train_steps}")



    def _train(self):
        loss_sum = 0
        n = 0
        
        with torch.enable_grad(), tqdm(total=self.train_steps) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_dataloader.train_loader):
                origin_y = origin_y.to(self.device)
                self.model_optim.zero_grad()
                        
                batch_size = batch_x.size(0)
                batch_x = batch_x.to(self.device, dtype=torch.float32)
                batch_y = batch_y.to(self.device, dtype=torch.float32)
                stamp = self.time_stamp.expand(batch_size, -1)

                x = batch_x.permute(0, 2, 1).unsqueeze(-1) #  (B, N, T, 1)
                y = batch_y.permute(0, 2, 1).unsqueeze(-1)  #  (B, N, O, 1)

                x = x.repeat(2, 1, 1, 1)
                stamp = stamp.repeat(2, 1)
                y = y.repeat(2, 1, 1, 1)
                bs = y.shape[0]
                y_pred, loss = self.model(x, stamp, y, self.current_epoch)
                y_pred1 = y_pred[:bs//2, :, :, :]
                y_pred2 = y_pred[bs//2:, :, :, :]
                r_loss = torch.functional.F.l1_loss(y_pred1, y_pred2)
                r_loss = r_loss * self.opt.r
                loss = loss + r_loss

                self.model_optim.zero_grad()
                loss.backward()
                self.model_optim.step()
                loss_sum += loss.item()
                n += 1
                
                progress_bar.update(batch_x.size(0))

            # scheduler.step()
            return loss_sum/n
                

    def _init_model(self):
        opt = DefaultConfig()
        opt.T4N['step'] = self.T4N_steps
        
        self.opt = opt
        
        opt.n_route = self.dataset.num_features
        opt.n_hist = self.windows
        opt.n_pred = self.pred_len
        
        opt.dis_mat =  torch.tensor(self.dataset.adj).to(self.device).float()
        n_route = opt.n_route
        n_his = opt.n_his
        n_pred = opt.n_pred
        enc_spa_mask = torch.ones(1, 1, n_route, n_route).to(self.device)
        enc_tem_mask = torch.ones(1, 1, n_his, n_his).to(self.device)
        dec_slf_mask = torch.tril(torch.ones(
            (1, 1, n_pred + 1, n_pred + 1)), diagonal=0).to(self.device)
        dec_mul_mask = torch.ones(1, 1, n_pred + 1, n_his).to(self.device)

        self.time_stamp = torch.range(1, self.windows).int().to(self.device)
        
        
        self.model = TPGNN(opt, enc_spa_mask, enc_tem_mask, dec_slf_mask, dec_mul_mask).to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        
        
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        stamp = self.time_stamp.expand(batch_size, -1)
        src = batch_x.permute(0, 2, 1).unsqueeze(-1) #  (B, N, T, 1)
        label = batch_y.permute(0, 2, 1).unsqueeze(-1)  #  (B, N, O, 1)
        enc_input = self.model.src_pro(src, stamp)

        # enc_input = model.enc_exp(src)
        # enc_input = model.enc_spa_enco(enc_input)
        # enc_input = model.enc_tem_enco(enc_input)
        stamp_emb = self.model.stamp_emb(stamp)
        enc_output = self.model.encoder(enc_input, stamp_emb)
        
        trg = torch.zeros(label.shape).to(self.device)
        for i in range(self.pred_len):
            dec_input = self.model.trg_pro(trg, enc_output)
            dec_output = self.model.decoder(dec_input, enc_output)
            dec_output = self.model.dec_rdu(dec_output)
            trg[:, :, i, :] = dec_output[:, :, i, :]
        outputs = trg.squeeze(-1).transpose(1,2)

        # outputs = list()
        # for i in range(self.pred_len):
        #     dec_input = self.model.trg_pro(trg, enc_output)
        #     dec_output = self.model.decoder(dec_input, enc_output)
        #     dec_output = self.model.dec_rdu(dec_output)
        #     outputs.append(dec_output[:, :, i, :])
        # outputs = torch.stack(outputs, dim=-2)
        # outputs = outputs.squeeze(-1).transpose(1,2)
        
        
        # outputs, loss = self.model(src, stamp, label, 0)  # (horizon, batch_size, num_sensor * output)
        return outputs, batch_y



def main():
    exp = TPGNNExperiment(
        dataset_type="DummyDatasetGraph",
        data_path="./data",
        optm_type="Adam",
        horizon=1,
        pred_len=3,
        batch_size=16,
        device="cuda:0",
        windows=12,
        T4N_steps=1,
    )
    # exp.config_wandb(
    #     "project",
    #     "name"
    # )
    exp.run()



def cli():
    import fire
    fire.Fire(TPGNNExperiment)


if __name__ == "__main__":
    cli()

