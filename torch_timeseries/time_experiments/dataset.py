from torch_timeseries.data.scaler import MaxAbsScaler, Scaler
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.utils.timefeatures import time_features
from torch.utils.data import Dataset

class MultiStepTimeFeatureSet(Dataset):
    def __init__(self, dataset: TimeSeriesDataset, scaler: Scaler, time_enc=0, window: int = 168, horizon: int = 3, steps: int = 2, freq=None, scaler_fit=True):
        self.dataset = dataset
        self.window = window
        self.horizon = horizon
        self.steps = steps
        self.time_enc = time_enc
        self.scaler = scaler
        
        self.num_features = self.dataset.num_features
        self.length = self.dataset.length
        if freq is None:
            self.freq = self.dataset.freq
        else:
            self.freq = freq
        if scaler_fit:
            self.scaler.fit(self.dataset.data)
        self.scaled_data = self.scaler.transform(self.dataset.data)
        self.date_enc_data = time_features(
            self.dataset.dates, self.time_enc, self.freq)
        assert len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1 > 0, "Dataset is not long enough!!!"


    def transform(self, values):
        return self.scaler.transform(values)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)

    def __getitem__(self, index):
        # x : (B, T, N)
        # y : (B, O, N)
        # x_date_enc : (B, T, D)
        # y_date_eDc : (B, O, D)
        if isinstance(index, int):
            x = self.scaled_data[index:index+self.window]
            x_date_enc = self.date_enc_data[index:index+self.window]
            scaled_y = self.scaled_data[self.window + self.horizon - 1 +
                                 index:self.window + self.horizon - 1 + index+self.steps]
            y = self.dataset.data[self.window + self.horizon - 1 +
                                 index:self.window + self.horizon - 1 + index+self.steps]
            y_date_enc = self.date_enc_data[self.window + self.horizon -
                                            1 + index:self.window + self.horizon - 1 + index+self.steps]
            return x, scaled_y,y, x_date_enc, y_date_enc
        else:
            raise TypeError('Not surpported index type!!!')

    def __len__(self):
        return len(self.dataset) - self.window - self.horizon + 1 - self.steps + 1
