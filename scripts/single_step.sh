#!/bin/bash

# 24G 可以跑 2 个ETT + 一个小 

# running
# t1 : ./scripts/single_step.sh "3 6 12" "ETTm1 ETTm2"  "cuda:2" HSTGNN

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

horizons=($1)     
datasets=($2)     
device=$3         
model=$4        
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for horizon in "${horizons[@]}"
    do
        echo "Running with dataset = $dataset and horizon = $horizon and window= $window"
        # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=HSTGNN --name="HSTGNN" runs --seeds='[42,233,666,19971203,19980224]'
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$model.py --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon="$horizon" --windows $window --epochs=20  runs --seeds='[42,233]'
    done
done

echo "All runs completed."
