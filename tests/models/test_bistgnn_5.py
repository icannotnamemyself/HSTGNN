import pytest
import torch
from torch_timeseries.datasets.dataset import TimeSeriesDataset,TimeSeriesStaticGraphDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.nn.embedding import PositionalEmbedding, TemporalEmbedding, FixedEmbedding, TimeFeatureEmbedding, DataEmbedding
from torch_timeseries.models.BiSTGNNv5 import BiSTGNNv5
from torch_timeseries.data.scaler import *
import os
import random


def test_bistgnn(dummy_dataset_time: TimeSeriesDataset):
    window = 12
    device = 'cuda:1'
    dataloader = ChunkSequenceTimefeatureDataLoader(dummy_dataset_time,
                                                    scaler=StandarScaler(device=device),
                                                    window = window,
                                                    horizon=1,
                                                    steps=1
                                                    )
    # 5 for minutes temporal embedding
    model = BiSTGNNv5(
            window,
            dummy_dataset_time.num_features,
            temporal_embed_dim=5,
            graph_conv_type='han',
    )
    model = model.to(device)
    for i, (
        batch_x,
        batch_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(dataloader.train_loader):
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(device).float()
        batch_y_date_enc = batch_y_date_enc.to(device).float()
        batch_x = batch_x.transpose(1,2)
        outputs = model(batch_x, batch_x_date_enc)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        # return outputs.squeeze(1), batch_y.squeeze(1)
        
        
        
def test_bistgnn_multistep(dummy_dataset_time: TimeSeriesDataset):
    window = 16
    device = 'cuda:1'
    dataloader = ChunkSequenceTimefeatureDataLoader(dummy_dataset_time,
                                                    scaler=StandarScaler(device=device),
                                                    window = window,
                                                    horizon=1,
                                                    steps=24
                                                    )
    # 5 for minutes temporal embedding
    model = BiSTGNNv5(
            window,
            dummy_dataset_time.num_features,
            temporal_embed_dim=0,
            graph_conv_type='han',
            out_seq_len=24,
            tn_layers=2
            
    )
    model = model.to(device)
    for i, (
        batch_x,
        batch_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(dataloader.train_loader):
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(device).float()
        batch_y_date_enc = batch_y_date_enc.to(device).float()
        batch_x = batch_x.transpose(1,2)
        outputs = model(batch_x, batch_x_date_enc)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        # return outputs.squeeze(1), batch_y.squeeze(1)
        
        
def test_bistgnn_predefined(dummy_dataset_graph: TimeSeriesStaticGraphDataset):
    window = 16
    device = 'cuda:1'
    dataloader = ChunkSequenceTimefeatureDataLoader(dummy_dataset_graph,
                                                    scaler=StandarScaler(device=device),
                                                    window = window,
                                                    horizon=1,
                                                    steps=24
                                                    )
    # 5 for minutes temporal embedding
    model = BiSTGNNv5(
            window,
            dummy_dataset_graph.num_features,
            temporal_embed_dim=5,
            graph_conv_type='han',
            graph_build_type='predefined_adaptive',
            out_seq_len=24,
            predefined_NN_adj=dummy_dataset_graph.adj
            
    )
    model = model.to(device)
    for i, (
        batch_x,
        batch_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(dataloader.train_loader):
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_y = batch_y.to(device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(device).float()
        batch_y_date_enc = batch_y_date_enc.to(device).float()
        batch_x = batch_x.transpose(1,2)
        outputs = model(batch_x, batch_x_date_enc)  # torch.Size([batch_size, num_nodes])
        # single step prediction
        # return outputs.squeeze(1), batch_y.squeeze(1)