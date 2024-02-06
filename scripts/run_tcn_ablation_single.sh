#!/bin/bash

# 24G 可以跑 2 个ETT + 一个小 
# t1 : ./scripts/run.sh "3 6 " "ETTh2 ETTm1 ETTm2"  "cuda:0" 384
# ./scripts/run.sh "3 6 12 24" "ExchangeRate"  "cuda:0" 96
# ./scripts/run.sh "3 6 12 24" "SolarEnergy"  "cuda:0" 168
# ./scripts/run.sh "3 6 12 24" "Traffic"  "cuda:1" 168

# ./scripts/run.sh "3 6 12 24" "SolarEnergy  Electricity Traffic"  "cuda:1" 168 nlinear
# ./scripts/run.sh "3 6 12 24" "ILI"  "cuda:1" 48 nlinear
# ./scripts/run.sh "3 6 12 24" "SP500"  "cuda:1" 40 nlinear

# t1 : ./scripts/run.sh "12 24" "ETTh2 ETTm1 ETTm2"  "cuda:0" 384

# ./scripts/single_step_baseline.sh "3 6 12 24" "Traffic Electricity SolarEnergy" "cuda:1" 168 tsmixer
# ./scripts/single_step_baseline.sh "3 6 12 24" "ILI" "cuda:1" 48 tsmixer
# ./scripts/single_step_baseline.sh "3 6 12 24" "SP500" "cuda:1" 40 tsmixer
# ./scripts/single_step_baseline.sh "3 6 12 24" "SP500" "cuda:1" 40 mtgnn
# ./scripts/single_step_baseline.sh "3 6 12 24" "ILI" "cuda:1" 48 mtgnn

# ./scripts/run_tcn_ablation.sh "3 6 12 24" "ETTh1 ETTh2 ETTm1 ETTm2" "cuda:1" 384
# ./scripts/run_tcn_ablation.sh "3 6 12 24" "SolarEnergy Traffic Electicity Weather" "cuda:1" 168
# ./scripts/run_tcn_ablation.sh "3 6 12 24" "ILI" "cuda:1" 48
# ./scripts/run_tcn_ablation_single.sh "3 6 12 24" "ExchangeRate" "cuda:1" 96 
# ./scripts/run_tcn_ablation_single.sh "3 6 12 24" "Electricity Traffic" "cuda:1" 168


# 接收 horizon 和 datasets 作为参数
horizons=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$3          # 第三个参数，用于指定 device，比如 "cuda:0"
windows=$4          # 第三个参数，用于指定 device，比如 "cuda:0"
# baseline=$5

# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    # 然后循环 horizons
    for horizon in "${horizons[@]}"
    do
        echo "Running with dataset = $dataset and horizon = $horizon"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv3.py --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon="$horizon" --windows $windows --without_tn_module=True --epochs=100 config_wandb --project=BiSTGNN --model_type="TCN_ablation" --name="TCN_ablation" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
