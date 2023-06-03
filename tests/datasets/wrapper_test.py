import pytest
from torch_timeseries.datasets.wrapper import SingleStepWrapper, MultiStepFlattenWrapper, SingStepFlattenWrapper, MultiStepWrapper
from torch.utils.data import DataLoader, Subset, RandomSampler
import torch

def test_single_step_wrapper(dummy_dataset):
    to_loader_wrapper = SingleStepWrapper(dummy_dataset, 144, 3)
    assert (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all() and  (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[1][0][0] == [1] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[0][1] == [144 - 1 + 3] * dummy_dataset.num_features).all()



def test_multi_step_wrapper(dummy_dataset):
    to_loader_wrapper = MultiStepWrapper(dummy_dataset, 144, 3, steps=2)
    assert (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all() and  (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[1][0][0] == [1] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[0][1] == [[144 - 1 + 3] * dummy_dataset.num_features , [144 - 1 + 4] * dummy_dataset.num_features]  ).all()



def test_multi_step_flatten_wrapper(dummy_dataset):
    to_loader_wrapper = MultiStepFlattenWrapper(dummy_dataset, 144, 3)
    assert (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all() and  (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[1][0][0] == [1] * dummy_dataset.num_features).all()
    data = ([144 - 1 + 3] * dummy_dataset.num_features)
    data.extend([144 - 1 + 4] * dummy_dataset.num_features ) 
    assert (to_loader_wrapper[0][1] ==  data).all()


def test_single_step_flatten_wrapper(dummy_dataset):
    to_loader_wrapper = SingStepFlattenWrapper(dummy_dataset, 144, 3)
    assert (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all() and  (to_loader_wrapper[0][0][0] == [0] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[1][0][0] == [1] * dummy_dataset.num_features).all()
    assert (to_loader_wrapper[0][1] == [144 - 1 + 3] * dummy_dataset.num_features).all()


