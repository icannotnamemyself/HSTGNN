#!/bin/bash

# 24G 可以跑 2 个ETT + 一个小 

# running
# t1 : ./scripts/run.sh "3 6 " "ETTm1 ETTm2"  "cuda:2" 384
# t1 : ./scripts/run.sh "12 24 " "ETTh1 ETTh2"  "cuda:1" 384
# t1 : ./scripts/run.sh "12 24" "ETTm1 ETTm2"  "cuda:0" 384
# t1 : ./scripts/run.sh "3 6 " "ETTh1 ETTh2"  "cuda:1" 384
# ./scripts/run.sh "3 6 12 24" "SP500" "cuda:0" 40
# ./scripts/run.sh "3 6 12 24" "ExchangeRate" "cuda:0" 48
# ./scripts/run.sh "3 6" "ILI SP500 ExchangeRate" "cuda:0" 
# ./scripts/run.sh "6" "Traffic" "cuda:0" 
# ./scripts/run.sh "12" "Traffic" "cuda:0"  
# ./scripts/run.sh "3" "Traffic" "cuda:0"  
# ./scripts/run.sh "24" "Traffic" "cuda:0"  
# ./scripts/run.sh "3 6" "Weather" "cuda:1" 
# ./scripts/run.sh "12 24" "Weather" "cuda:0" 

# ./scripts/run.sh "24" "Electricity" "cuda:3"  
# ./scripts/run.sh "3" "Electricity" "cuda:0"  


# t1 : ./scripts/run.sh "3" "ETTh1 ETTh2 "  "cuda:0"
# t1 : ./scripts/run.sh "6" "ETTh1 ETTh2 "  "cuda:0"

# ./scripts/run.sh "3 6 12 24" "SolarEnergy"  "cuda:0" 168
# ./scripts/run.sh "3 6 12 24" "Traffic"  "cuda:1" 168
# ./scripts/run.sh "3 6 12 24" "Traffic"  "cuda:1" 168
# t1 : ./scripts/run.sh "24 12 6 3" "Electricity"  "cuda:5"
# t1 : ./scripts/run.sh "24" "ETTm1"  "cuda:0"


# t1 : ./scripts/run.sh "24 12" "ETTm2 ETTm1"  "cuda:2"
# t1 : ./scripts/run.sh "3 6" "ETTh2 ETTh1"  "cuda:0"
# t1 : ./scripts/run.sh "3 6 12 24" "ExchangeRate ILI SP500"  "cuda:2"

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
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for horizon in "${horizons[@]}"
    do
        echo "Running with dataset = $dataset and horizon = $horizon and window= $window"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
