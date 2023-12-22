#!/bin/bash

# 定义 horizon 的值
horizons=(3 6 12 24)

# 定义数据集
datasets=("ETTh2" "ETTm1" "ETTm2")

# 首先循环数据集
for dataset in "${datasets[@]}"
do
    # 然后循环 horizon
    for horizon in "${horizons[@]}"
    do
        echo "Running with dataset = $dataset and horizon = $horizon"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv3.py --dataset_type="$dataset" --device="cuda:0" --batch_size=32 --horizon="$horizon" --windows 384 --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv3 runs" --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
