#!/bin/bash

# 定义 horizon 的值
horizons=(3 6 12 24)

# 循环遍历每个 horizon 值
for horizon in "${horizons[@]}"
do
    echo "Running with horizon = $horizon"
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv3.py --dataset_type="ETTh2" --device="cuda:0" --batch_size=32 --horizon="$horizon" --windows 384 --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv3 runs" --seeds='[42,233,666,19971203,19980224]'
done

echo "All runs completed."
