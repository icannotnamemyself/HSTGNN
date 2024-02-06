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
# ./scripts/single_step_baseline.sh "3 6 12 24" "ILI" "cuda:1"  esg


# ./scripts/single_step_baseline.sh "3 6" "ETTm2" "cuda:1"  esg
# ./scripts/single_step_baseline.sh "12 24" "ETTm2" "cuda:0"  esg

# ./scripts/single_step_baseline.sh "3 6 12 24" "ExchangeRate ILI SP500" "cuda:4"  esg

# ./scripts/single_step_baseline.sh "3 6" "Electricity" "cuda:0"  film
# ./scripts/single_step_baseline.sh "12 24" "Electricity" "cuda:4"  film
# ./scripts/single_step_baseline.sh "3 6 12 24" "ILI SP500" "cuda:0"  film

# ./scripts/single_step_baseline.sh "12 " "Electricity" "cuda:1"  film
# ./scripts/single_step_baseline.sh "24" "Electricity" "cuda:2"  film

# ./scripts/single_step_baseline.sh "3 6 12 24" "ILI SP500" "cuda:2"  film

# ./scripts/single_step_baseline.sh "3 6 12 24" "Electricity" "cuda:4"  dlinear
# ./scripts/single_step_baseline.sh "3 6 12 24" "Electricity" "cuda:4"  dlinear
# ./scripts/single_step_baseline.sh "3 6" "Electricity ILI SP500" "cuda:4"  tsmixer
# ./scripts/single_step_baseline.sh "12 24" "Electricity ILI SP500" "cuda:4"  film

# ./scripts/single_step_baseline.sh "6" "SP500" "cuda:4"  crossformer

declare -A dataset_to_window_map
dataset_to_window_map["ETTm1"]=384
dataset_to_window_map["ETTm2"]=384
dataset_to_window_map["ETTh1"]=384
dataset_to_window_map["ETTh2"]=384
dataset_to_window_map["ExchangeRate"]=96
dataset_to_window_map["ILI"]=48
dataset_to_window_map["SP500"]=40
dataset_to_window_map["Traffic"]=168
dataset_to_window_map["Electricity"]=168
dataset_to_window_map["Weather"]=168
dataset_to_window_map["SolarEnergy"]=168

# 接收 horizon 和 datasets 作为参数
horizons=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$3          # 第三个参数，用于指定 device，比如 "cuda:0"
baseline=$4
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    # 然后循环 horizons
    for horizon in "${horizons[@]}"
    do
        echo "Running with dataset = $dataset and horizon = $horizon, window = $window"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/$baseline.py --dataset_type="$dataset" --device="$device" --batch_size=16 --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="baseline" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
