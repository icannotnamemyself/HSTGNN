#!/bin/bash

# 24G 可以跑 2 个ETT + 一个小 
# t1 : ./scripts/run.sh "3 6 " "ETTh2 ETTm1 ETTm2"  "cuda:0" 384
# ./scripts/run.sh "3 6 12 24" "ExchangeRate"  "cuda:0" 96
# ./scripts/run.sh "3 6 12 24" "SolarEnergy"  "cuda:0" 168
# ./scripts/run.sh "3 6 12 24" "Traffic"  "cuda:1" 168

# ./scripts/run.sh "3 6 12 24" "SolarEnergy  Electricity Traffic"  "cuda:1" 168 nlinear
# ./scripts/run.sh "3 6 12 24" "ILI"  "cuda:1" 48 nlinear
# ./scripts/run.sh "3 6 12 24" "SP500"  "cuda:1" 40 nlinear

# t1 : ./scripts/multi_step_baseline.sh "12 24" "ETTh2 ETTm1 ETTm2"  "cuda:0" esg

# ./scripts/multi_step_baseline.sh "3 6 12" "METR_LA" "cuda:0" esg
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS_BAY" "cuda:0" esg
# ./scripts/multi_step_baseline.sh "6" "PEMS04" "cuda:0" esg
# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:1" esg
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS07" "cuda:0" esg

# ./scripts/multi_step_baseline.sh "3 6 12" "METR_LA" "cuda:2" dcrnn
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS_BAY" "cuda:4" dcrnn


# ./scripts/multi_step_baseline.sh "6" "PEMS07" "cuda:0" dcrnn


# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:1" esg
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS_BAY" "cuda:4" dcrnn
# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:1" esg
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS04" "cuda:0" esg

# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS04" "cuda:3" dcrnn
# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS04" "cuda:4" dcrnn
# ./scripts/multi_step_baseline.sh "3" "PEMS07" "cuda:5" dcrnn
# ./scripts/multi_step_baseline.sh "6" "PEMS07" "cuda:4" dcrnn
# ./scripts/multi_step_baseline.sh "12" "PEMS07" "cuda:3" dcrnn


# ./scripts/multi_step_baseline.sh "3 6" "PEMS_BAY" "cuda:0" dcrnn
# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:0" dcrnn
# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:0" dcrnn


# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS04" "cuda:1" stgcn
# ./scripts/multi_step_baseline.sh "6 12" "METR_LA" "cuda:0" stgcn
# ./scripts/multi_step_baseline.sh "3" "PEMS07" "cuda:0" stgcn
# ./scripts/multi_step_baseline.sh "6" "PEMS07" "cuda:0" stgcn
# ./scripts/multi_step_baseline.sh "12" "PEMS07" "cuda:0" stgcn
# ./scripts/multi_step_baseline.sh "12" "PEMS_BAY" "cuda:0" gman

# ./scripts/multi_step_baseline.sh "6" "PEMS04" "cuda:1" gman


# ./scripts/multi_step_baseline.sh "3 6 12" "PEMS04" "cuda:0" dcrnn
# ./scripts/multi_step_baseline.sh "3 6 12" "METR_LA" "cuda:0" dcrnn

# ./scripts/multi_step_baseline.sh "3 6 12" "METR_LA" "cuda:1" stgcn


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
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/$baseline.py  --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12 --epochs=100 config_wandb --project=BiSTGNN --name="baseline" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
