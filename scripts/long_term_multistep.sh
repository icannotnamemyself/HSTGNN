#!/bin/bash


# running
# ./scripts/long_term_multistep.sh "720" "ExchangeRate"  "cuda:2" HSTGNN


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

pred_lens=($1)     
datasets=($2)     
device=$3         
model=$4        
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for pred_len in "${pred_lens[@]}"
    do
        echo "Running with dataset = $dataset and pred_len = $pred_len and window= $window"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$model.py --dataset_type="$dataset" --device="$device" --horizon=1 --batch_size=16 --pred_len="$pred_len" --windows $window --epochs=100  runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
