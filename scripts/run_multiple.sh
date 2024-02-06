#!/bin/bash

# 24G 可以跑 2 个ETT + 一个小 
# t1 : ./scripts/run.sh "3 6 " "ETTh2 ETTm1 ETTm2"  "cuda:0" 384
# ./scripts/run_multiple.sh "12 6 3" "PEMS03 PeMS_D7 PEMS08" "cuda:0" 12
# ./scripts/run_multiple.sh "6" "METR_LA PEMS04 PEMS_BAY PEMS07 PEMS03 PeMS_D7 PEMS08 " "cuda:1" 12
# ./scripts/run_multiple.sh "3" "METR_LA PEMS04  PEMS_BAY PEMS07 PEMS03 PeMS_D7 PEMS08" "cuda:2" 12

# running
# ./scripts/run_multiple.sh "3 6 12"  "METR_LA PEMS04" "cuda:2" 12
# ./scripts/run_multiple.sh "3 6 12"  "METR_LA PEMS04" "cuda:2" 12

# ./scripts/run_multiple.sh "3" "PEMS04 METR_LA" "cuda:0" 12 
# ./scripts/run_multiple.sh "6" "PEMS04 METR_LA" "cuda:1" 12 
# ./scripts/run_multiple.sh "12" "PEMS04 METR_LA" "cuda:2" 12 

# ./scripts/run_multiple.sh "3" "PEMS07" "cuda:0" 12 
# ./scripts/run_multiple.sh "6" "PEMS07" "cuda:0" 12 
# ./scripts/run_multiple.sh "12" "PEMS07" "cuda:0" 12 

# ./scripts/run_multiple.sh "3" "PEMS_BAY" "cuda:1" 12 
# ./scripts/run_multiple.sh "6" "PEMS_BAY" "cuda:0" 12 
# ./scripts/run_multiple.sh "12" "PEMS_BAY" "cuda:0" 12 

# ./scripts/run_multiple.sh "12" "PEMS_BAY PEMS07" "cuda:0" 12
# ./scripts/run_multiple.sh "6" "METR_LA PEMS04" "cuda:1" 12 
# ./scripts/run_multiple.sh "6" "PEMS_BAY PEMS07" "cuda:1" 12 
# ./scripts/run_multiple.sh "3" "METR_LA PEMS04" "cuda:2" 12
# ./scripts/run_multiple.sh "12" "PEMS_BAY" "cuda:0" 12


# ./scripts/run_multiple.sh "3 6" "PEMS_BAY" "cuda:1" 12
# ./scripts/run_multiple.sh "" "PEMS_BAY" "cuda:0" 12

# ./scripts/run_multiple.sh "3" "PEMS07" "cuda:1" 12
# ./scripts/run_multiple.sh "6" "PEMS07" "cuda:0" 12
# ./scripts/run_multiple.sh "3 6 12" "PEMS07" "cuda:0" 12

# ./scripts/run_multiple.sh "3 6 12" "METR_LA" "cuda:0" 12
# ./scripts/run_multiple.sh "3 6 12" "PEMS04" "cuda:0" 12

# 接收 horizon 和 datasets 作为参数
pred_lens=($1)      # 第一个参数，期望为类似 "3 6 12 24" 这样的格式
datasets=($2)      # 第二个参数，期望为类似 "ETTh1 ETTh2 ETTm1" 这样的格式
device=$3          # 第三个参数，用于指定 device，比如 "cuda:0"
windows=$4          # 第三个参数，用于指定 device，比如 "cuda:0"

# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    # 然后循环 horizons
    for pred_len in "${pred_lens[@]}"
    do
        echo "Running with dataset = $dataset and pred_len = $pred_len"
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/cli/hstgnnv7.py  --invtrans_loss=True --dataset_type="$dataset" --latent_dim=32 --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows $windows --epochs=100 config_wandb --project=BiSTGNN --name="HSTGNNv7" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
