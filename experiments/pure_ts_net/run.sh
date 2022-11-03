#!/bin/sh
export DATASET=exchange_rate
export NUM_NODES=8
export ROOT=/notebooks/MTGNN

export HORIZON=3
export PYTHONUNBUFFERED=1
export PYTHONPATH=/notebooks/pytorch_timeseries
python ./net_train_single_step.py  --horizon ${HORIZON} --data ${ROOT}/data/${DATASET}.txt  --save model/${DATASET}_horizon_${HORIZON}.pt --device cuda:2 --gcn_true False --residual_layer False --num_nodes ${NUM_NODES}  --epochs 30 --skip_layer False --skip_channels 16