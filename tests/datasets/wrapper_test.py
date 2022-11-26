import pytest
from torch_timeseries.datasets.wrapper import SingleStepWrapper, MultiStepFlattenWrapper, SingStepFlattenWrapper, MultiStepWrapper



def test_single_step_wrapper(dummy_dataset):
    to_loader_wrapper = SingleStepWrapper(dummy_dataset, 144, 3)
    assert (to_loader_wrapper[0][0][0] == [0] * 100).all() and  (to_loader_wrapper[0][0][0] == [0] * 100).all()
    assert (to_loader_wrapper[1][0][0] == [1] * 100).all()



def test_multi_step_wrapper(dummy_dataset):
    to_loader_wrapper = MultiStepWrapper(dummy_dataset, 144, 3, steps=2)
    assert (to_loader_wrapper[0][0][0] == [0] * 100).all() and  (to_loader_wrapper[0][0][0] == [0] * 100).all()
    assert (to_loader_wrapper[0][1][0] == [146] * 100).all()
    assert (to_loader_wrapper[0][1][1] == [147] * 100).all()



def test_multi_step_flatten_wrapper(dummy_dataset):
    to_loader_wrapper = MultiStepFlattenWrapper(dummy_dataset, 144, 3)
    assert to_loader_wrapper[0][0].shape[0] == 144* 100 
    assert (to_loader_wrapper[0][1][0] == [146] * 100).all()
    assert (to_loader_wrapper[0][1][1] == [147] * 100).all()


def test_single_step_flatten_wrapper(dummy_dataset):
    to_loader_wrapper = SingStepFlattenWrapper(dummy_dataset, 144, 3)
    assert (to_loader_wrapper[0][1] == [146] * 100).all()
    assert (to_loader_wrapper[2][1] == [148] * 100).all()


