#!/bin/bash

# ./scripts/gcn_ablation_multiple.sh "12" "METR_LA" "graphsage"  "cuda:0"  
# ./scripts/gcn_ablation_multiple.sh "12" "PEMS04" "graphsage"  "cuda:1"  
# ./scripts/gcn_ablation_multiple.sh "12" "METR_LA" "fagcn"  "cuda:0"  
# ./scripts/gcn_ablation_multiple.sh "12" "PEMS04 " "fagcn"  "cuda:1"  
# ./scripts/gcn_ablation_multiple.sh "12" "METR_LA" "han"  "cuda:0"  


declare -A dataset_to_window_map

dataset_to_window_map["PEMS_BAY"]=12
dataset_to_window_map["METR_LA"]=12
dataset_to_window_map["PEMS04"]=12
dataset_to_window_map["PEMS07"]=12


# 接收 horizon 和 datasets 作为参数
pred_lens=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
gcn_models=($3)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$4          # 第三个参数，用于指定 device，比如 "cuda:0"
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for pred_len in "${pred_lens[@]}"
    do

        for gcn_model in "${gcn_models[@]}"
            do
                echo "Running with dataset = $dataset and horizon = $horizon and window= $window"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py  --invtrans_loss=True --dataset_type="$dataset" --latent_dim=32 --device="$device"  --gcn_type=$gcn_model --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'

                # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --gcn_type=$gcn_model --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
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
