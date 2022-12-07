


def test_mtcsg():
    from torch_timeseries.nn.mtcsg import MTCSG
    import torch
    model = MTCSG(8, 130, middle_channel=16, seq_out=1,causal_aggr_layers=1)
    data = torch.randn(64, 1, 8, 130)
    model(data)
