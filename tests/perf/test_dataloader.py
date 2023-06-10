






import time
from torch_timeseries.data.scaler import MaxAbsScaler
from torch_timeseries.datasets import ExchangeRate
from torch_timeseries.datasets.dataloader import ChunkSequenceTimefeatureDataLoader
from torch_timeseries.datasets.traffic import Traffic


def test_dataloader():
    windows= 300
    horizon =3
    pred_len =1 
    batch_size= 1024
    num_worker = 10
    dataset = Traffic(root='./data')
    scaler = MaxAbsScaler()
    dataloader = ChunkSequenceTimefeatureDataLoader(
        dataset,
        scaler,
        window=windows,
        horizon=horizon,
        steps=pred_len,
        scale_in_train=False,
        shuffle_train=True,
        freq="h",
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.2,
        num_worker=num_worker,
    )
    
    epochs = 3

    cpu_data_load_time_start = time.time()
    for j in range(epochs):
        for i, (
            batch_x,
            batch_y,
            batch_x_date_enc,
            batch_y_date_enc,
        ) in enumerate(dataloader.train_loader):
            continue
    cpu_data_load_time_end = time.time()

    print(f"total data load time: {cpu_data_load_time_end  - cpu_data_load_time_start}")


