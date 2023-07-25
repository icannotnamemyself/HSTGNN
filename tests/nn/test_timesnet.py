import pytest
import torch
from torch_timeseries.data.scaler import StandarScaler
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceSplitter
from torch_timeseries.datasets.wrapper import MultiStepTimeFeatureSet
from torch_timeseries.nn.embedding import PositionalEmbedding, TemporalEmbedding, FixedEmbedding, TimeFeatureEmbedding, DataEmbedding
from torch_timeseries.models.TimesNet import TimesNet
from tqdm import tqdm

def test_timesnet(dummy_dataset_time: TimeSeriesDataset):
    torch.set_default_tensor_type(torch.FloatTensor)
    batch_size =  32
    dataset = MultiStepTimeFeatureSet(
        dummy_dataset_time,scaler=StandarScaler(), window=7, horizon=3, steps=3)
    srs = SequenceSplitter(batch_size=batch_size)
    pred_len = 3
    label_len = 2
    assert pred_len >= label_len
    model = TimesNet(enc_in=dummy_dataset_time.num_features, c_out=dummy_dataset_time.num_features, pred_len=pred_len)
    train_loader, val_loader, test_loader = srs(dataset)
    
    with tqdm( total=len(train_loader)) as progress_bar:
        for batch_x, batch_y, batch_x_date_enc, batch_y_date_enc in train_loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_date_enc = batch_x_date_enc.float()
            batch_y_date_enc = batch_y_date_enc.float()
            
            
            dec_inp_pred = torch.zeros([batch_x.size(0),  pred_len, dummy_dataset_time.num_features])
            dec_inp_label = batch_x[:, -label_len:, :]
            dec_inp = torch.cat([dec_inp_label,dec_inp_pred], dim=1)
            
            dec_inp_date_enc = torch.cat([batch_x_date_enc[:, -label_len:, :],batch_y_date_enc], dim=1)
            outputs = model(batch_x, batch_x_date_enc, dec_inp, dec_inp_date_enc)
            assert outputs[:, -pred_len:, :].shape == batch_y.shape

            progress_bar.update(1)