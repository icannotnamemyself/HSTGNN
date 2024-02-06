#!/bin/bash

# we run our model in dataset using different gcn layers
# ./scripts/latent_dim_ablation.sh "12" "METR_LA " "16 128"  "cuda:2" 
# ./scripts/latent_dim_ablation.sh "12" "METR_LA " "256"  "cuda:2" 
# ./scripts/latent_dim_ablation.sh "12" "METR_LA " "64"  "cuda:2" 
# ./scripts/latent_dim_ablation.sh "12" "METR_LA " "256"  "cuda:0" 
# ./scripts/latent_dim_ablation.sh "12" "METR_LA" "64"  "cuda:1" 


declare -A dataset_to_window_map

dataset_to_window_map["PEMS_BAY"]=12
dataset_to_window_map["METR_LA"]=12
dataset_to_window_map["PEMS04"]=12
dataset_to_window_map["PEMS07"]=12


# 接收 horizon 和 datasets 作为参数
pred_lens=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
latent_dims=($3)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$4          # 第三个参数，用于指定 device，比如 "cuda:0"
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for pred_len in "${pred_lens[@]}"
    do

        for latent_dim in "${latent_dims[@]}"
            do
                echo "Running with dataset = $dataset and window= $window"
        CUDA_DEVICE_ORDER=PCI_BUS_ID  python3 ./torch_timeseries/cli/hstgnnv7.py  --invtrans_loss=True --dataset_type="$dataset" --latent_dim=$latent_dim --device="$device"  --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'

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
