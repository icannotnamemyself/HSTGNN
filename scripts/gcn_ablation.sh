#!/bin/bash

# we run our model in dataset using different gcn layers


# t1 : ./scripts/gcn_ablation.sh "24" "ETTh1" "graphsage" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "ETTh1" "fagcn" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "ETTh1" "han" "cuda:0"

# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "graphsage" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "fagcn" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "han" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "Weather" "han" "cuda:0"

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
# gcn_models=("graphsage" "fagcn" "han" "hgt")


# 接收 horizon 和 datasets 作为参数
horizons=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
gcn_models=($3)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$4          # 第三个参数，用于指定 device，比如 "cuda:0"
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for horizon in "${horizons[@]}"
    do

        for gcn_model in "${gcn_models[@]}"
            do
                echo "Running with dataset = $dataset and horizon = $horizon and window= $window"
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --gcn_type=$gcn_model --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
            done
    done
done


# # 接收 horizon 和 datasets 作为参数
# horizons=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
# datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
# device=$3          # 第三个参数，用于指定 device，比如 "cuda:0"
# # 首先循环 datasets
# for dataset in "${datasets[@]}"
# do
#     window=${dataset_to_window_map[$dataset]}
#     for horizon in "${horizons[@]}"
#     do
#         echo "Running with dataset = $dataset and horizon = $horizon and window= $window"
#         CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
#     done
# done

echo "All runs completed."
