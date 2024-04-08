#!/bin/bash

# ./scripts/multi_step.sh "3 6 12" "PEMS07" "cuda:0" ESG
# ./scripts/multi_step.sh "3 6 12" "METR_LA" "cuda:0" DCRNN


# 接收 horizon 和 datasets 作为参数
pred_lens=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$3          # 第三个参数，用于指定 device，比如 "cuda:0"
baseline=$4          # 第三个参数，用于指定 device，比如 "cuda:0"

# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    # 然后循环 horizons
    for pred_len in "${pred_lens[@]}"
    do
        echo "Running with dataset = $dataset and pred_len = $pred_len, in window 12"
        # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/$baseline.py  --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12 --epochs=100 config_wandb --project=BiSTGNN --name="baseline" runs --seeds='[42,233,666,19971203,19980224]'
        # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$baseline.py --data_path="/notebooks/pytorch_timeseries/data" --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12 --epochs=100 config_wandb --project=BiSTGNN --name="baseline" runs --seeds='[42,233,666,19971203,19980224]'
        # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$baseline.py --data_path="/notebooks/pytorch_timeseries/data" --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12 --epochs=100 runs --seeds='[42,233,666,19971203,19980224]'
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$baseline.py --data_path="/notebooks/pytorch_timeseries/data" --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12  --epochs=20 runs --seeds='[42]'
    done
done

echo "All runs completed."
