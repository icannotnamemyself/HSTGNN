from torch_timeseries.experiments.stgcn_experiment import STGCNExperiment
import pandas as pd

device = 'cuda:0'
exp = STGCNExperiment(
    dataset_type="PEMS_BAY", # PEMS_BAY  PEMS04 PEMS07 METR_LA
    batch_size=32,
    horizon=1,
    invtrans_loss=True,
    pred_len=12,
    data_path='./data',
    save_dir='./results/',
    windows=12,
    device=device,
)
# # 42,233,666,19971203,19980224
seed = 42
exp._setup_run(seed)
if exp._check_run_exist(seed):
    exp._resume_run(seed)
# exp.run(seed)
# exp._resume_from('/notebooks/pytorch_timeseries/results/runs/MTGNN/PEMS_BAY/w12h1s12/ae9ae336bb8798c76d91d84448660ac8')

import torch
import time
total_time = 0
i = 0
for x, y, batch_origin_y, x_date_enc, y_date_enc in zip(xs, ys, boys, xdes, ydes):
    start = time.time()
    predicted = exp._process_one_batch(x, y, x_date_enc, y_date_enc)# exp.model(batch_x.transpose(1,2), batch_x_date_enc)
    end = time.time()
    total_time = total_time +  (end - start)
    i = i+ 1
    if i == 100:
        break
print(f" average inference time: {(total_time)/i}")
print(i)
# batch_x, batch_y,batch_origin_y, batch_x_date_enc, batch_y_date_enc