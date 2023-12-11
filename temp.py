from torch_geometric_temporal.dataset import PedalMeDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = PedalMeDatasetLoader()

dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

for time, snapshot in enumerate(train_dataset):
    import pdb;pdb.set_trace()