#!/bin/bash

# we run our model in dataset using different gcn layers
# ./scripts/gcn_layers_ablation.sh "12" "METR_LA " "1"  "cuda:0" 
# ./scripts/gcn_layers_ablation.sh "12" "METR_LA " "2"  "cuda:5" 
# ./scripts/gcn_layers_ablation.sh "12" "METR_LA " "3"  "cuda:2" 
# ./scripts/gcn_layers_ablation.sh "12" "METR_LA" "4"  "cuda:3"


# ./scripts/gcn_layers_ablation.sh "12" "METR_LA" "3 4"  "cuda:0"


# ./scripts/gcn_layers_ablation.sh "12" "METR_LA " "5"  "cuda:4" 
# ./scripts/gcn_layers_ablation.sh "12" "PEMS_BAY " "2"  "cuda:0" 
# ./scripts/gcn_layers_ablation.sh "6" "PEMS_BAY " "2"  "cuda:1" 
# ./scripts/gcn_layers_ablation.sh "3" "PEMS_BAY " "2"  "cuda:1" 


# t1 : ./scripts/gcn_ablation.sh "24" "ETTh1" "fagcn" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "ETTh1" "han" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "graphsage" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "fagcn" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "han" "cuda:0"
# t1 : ./scripts/gcn_ablation_multiple.sh "12" "PEMS_BAY PEMS04 METR_LA" "han" "cuda:0"
# t1 : ./scripts/gcn_ablation.sh "24" "SP500 ILI ExchangeRate ETTh2 ETTm1 ETTm2" "han" "cuda:0"
# ./scripts/run_multiple.sh "12" "PEMS_BAY " "cuda:0" 12
# ./scripts/run_multiple.sh "12" "METR_LA " "cuda:0" 12
# ./scripts/run_multiple.sh "12" "PEMS07 " "cuda:0" 12
# ./scripts/run_multiple.sh "12" "PEMS04 " "16 32 64 128"  "cuda:1"  


declare -A dataset_to_window_map

dataset_to_window_map["PEMS_BAY"]=12
dataset_to_window_map["METR_LA"]=12
dataset_to_window_map["PEMS04"]=12
dataset_to_window_map["PEMS07"]=12


# 接收 horizon 和 datasets 作为参数
pred_lens=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
gcn_layers=($3)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$4          # 第三个参数，用于指定 device，比如 "cuda:0"
# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    window=${dataset_to_window_map[$dataset]}
    for pred_len in "${pred_lens[@]}"
    do

        for gcn_layer in "${gcn_layers[@]}"
            do
                echo "Running with dataset = $dataset and window= $window and gcn_layer = $gcn_layer"
        CUDA_DEVICE_ORDER=PCI_BUS_ID  python3 ./torch_timeseries/cli/hstgnnv7.py  --invtrans_loss=True --dataset_type="$dataset" --latent_dim=32 --gcn_layers=$gcn_layer --device="$device"  --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'

                # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py --dataset_type="$dataset" --device="$device" --batch_size=32 --gcn_type=$gcn_model --horizon="$horizon" --windows $window --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
            done
    done
done



echo "All runs completed."
