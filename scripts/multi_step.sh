#!/bin/bash

# ./scripts/multi_step.sh "3 6 12" "PEMS07" "cuda:0" ESG
# ./scripts/multi_step.sh "3 6 12" "METR_LA" "cuda:0" DCRNN


pred_lens=($1)      
datasets=($2)     
device=$3          
baseline=$4          

# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    # 然后循环 horizons
    for pred_len in "${pred_lens[@]}"
    do
        CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/$baseline.py --data_path="/notebooks/pytorch_timeseries/data" --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 12 --epochs=100 config_wandb --project=BiSTGNN --name="baseline" runs --seeds='[42,233,666,19971203,19980224]'
    done
done

echo "All runs completed."
