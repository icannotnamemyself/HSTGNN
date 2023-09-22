import pytest
import torch
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.nn.embedding import PositionalEmbedding, TemporalEmbedding, FixedEmbedding, TimeFeatureEmbedding, DataEmbedding
from torch_timeseries.models.BiSTGNNv2 import BiSTGNNv2 
from torch_timeseries.data.scaler import *
import os
import random

def reproducible(seed):
    # for reproducibility
    # torch.set_default_dtype(torch.float32)
    print("torch.get_default_dtype()", torch.get_default_dtype())
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determinstic = True

def test_bistgnn(dummy_dataset_time: TimeSeriesDataset):
    
    reproducible(0)
    
    window = 128
    device = 'cuda:1'
    dataloader = ChunkSequenceTimefeatureDataLoader(dummy_dataset_time,
                                                    scaler=StandarScaler(device=device),
                                                    window = window,
                                                    horizon=1,
                                                    steps=1
                                                    )
    # 5 for minutes temporal embedding
    model = BiSTGNNv2(
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





def test_tntcn_graph_build(dummy_dataset_time: TimeSeriesDataset):
    window = 16
    device = 'cuda:1'
    dataloader = ChunkSequenceTimefeatureDataLoader(dummy_dataset_time,
                                                    scaler=StandarScaler(device=device),
                                                    window = window,
                                                    horizon=1,
                                                    steps=1
                                                    )
    
    model = TNTCN(
            n_nodes=dummy_dataset_time.num_features,
            input_seq_len=window,
            pred_horizon=1,
            multi_pred=False,
            graph_build_type='weighted_random_clip',
            without_gc=True,
            output_module='tcn',
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
        outputs = model(batch_x)  # torch.Size([batch_size, num_nodes])
        
        # single step prediction
        # return outputs.squeeze(1), batch_y.squeeze(1)

