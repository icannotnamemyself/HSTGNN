#!/bin/bash

# ./scripts/hstgnn.sh "12" "PEMS04" "cuda:0" hetero
# ./scripts/hstgnn.sh "12" "PEMS04" "cuda:0" homo
# ./scripts/hstgnn.sh "12" "PEMS04" "cuda:0" all
# ./scripts/hstgnn.sh "12" "PEMS04" "cuda:1" no
# ./scripts/hstgnn.sh "12" "PEMS04" "cuda:0" all MAGNN


# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" all HSTAttn 2


# ./scripts/hstgnn.sh "12" "PEMS_BAY" "cuda:0" hetero
# ./scripts/hstgnn.sh "12" "PEMS_BAY" "cuda:0" homo
# ./scripts/hstgnn.sh "12" "PEMS_BAY" "cuda:0" all
# ./scripts/hstgnn.sh "12" "PEMS_BAY" "cuda:0" no


# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" all
# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" no


# ./scripts/hstgnn.sh "12" "METR_LA" "cuda:0" hetero
# ./scripts/hstgnn.sh "12" "METR_LA" "cuda:0" homo
# ./scripts/hstgnn.sh "12" "METR_LA" "cuda:0" all
# ./scripts/hstgnn.sh "12" "METR_LA" "cuda:0" no


# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" hetero
# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" homo
# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" all HAN
# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" no HGT
# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" no MAGNN

# ./scripts/hstgnn.sh "12" "PEMS07" "cuda:0" no MAGNN



pred_lens=($1)      
datasets=($2)     
device=$3          
conv_type=$4          
gcn_type=$5     
gcn_layers=$6     
# latent_dim=$5          
# gcn_layers=$5          

# 首先循环 datasets
for dataset in "${datasets[@]}"
do
    # 然后循环 horizons
    for pred_len in "${pred_lens[@]}"
    do
        if [ $conv_type == 'no' ]
        then
            CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/HSTGNN.py --without_tn_module=True --data_path="/notebooks/pytorch_timeseries/data" --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows 168 --epochs=40  --patience=40 runs --seeds='[42,233,666,19971203,19980224]'
        else
            CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/experiments/HSTGNN.py  --conv_type=$conv_type --gcn_type=$gcn_type  --data_path="/notebooks/pytorch_timeseries/data" --gcn_layers=$gcn_layers --latent_dim=16 --invtrans_loss=True --dataset_type="$dataset" --device="$device" --batch_size=16 --horizon=1 --pred_len="$pred_len" --windows 168 --epochs=40 --patience=40 runs --seeds='[42,233,666,19971203,19980224]'
        fi

    done
done

echo "All runs completed."
