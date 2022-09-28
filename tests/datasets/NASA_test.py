from torch_timeseries.datasets import NASA




def test_NASA():
    nasa = NASA('./data', name="SMAP")
    print(nasa.download.__qualname__)
    pass