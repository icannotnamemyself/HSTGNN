import pytest
import torch
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.nn.embedding import PositionalEmbedding, TemporalEmbedding, FixedEmbedding, TimeFeatureEmbedding, DataEmbedding
from torch_timeseries.utils.timefeatures import time_features


def test_data_embedding(dummy_dataset_time):
    torch.set_default_tensor_type(torch.DoubleTensor)
    dataset = MultiStepTimeFeatureSet(
        dummy_dataset_time, window=7, horizon=3, steps=3)
    srs = SequenceSplitter()
    train_loader, val_loader, test_loader = srs(dataset)
    for x, y, x_date_enc, y_date_enc in train_loader:
        embed = DataEmbedding(2, 512, embed_type='fixed', freq='h')
        output = embed(x, x_date_enc)
        assert output.shape == (srs.batch_size, dataset.window, embed.d_model)

