import torch
from torch import Tensor
from typing import Generic, TypeVar, Union

import pandas as pd
import numpy as np
import torch

StoreType = TypeVar(
    "StoreType", bound=Union[pd.DataFrame, np.ndarray, torch.Tensor]
)  # Type variable for input and output data




class Scaler(Generic[StoreType]):
    def fit(self, data: StoreType) -> None:
        """
        Fit the Scaler  to the input dataset.

        Args:
            data:
                The input dataset to fit the Scaler  to.
        Returns:
            None.
        """
        raise NotImplementedError()

    def transform(self, data: StoreType) -> StoreType:
        """
        Transform the input dataset using the Scaler .

        Args:
            data:
                The input dataset to transform using the Scaler .
        Returns:
            The transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()

    def inverse_transform(self, data: StoreType) -> StoreType:
        """
        Perform an inverse transform on the input dataset using the Scaler .

        Args:
            data:
                The input dataset to perform an inverse transform on.
        Returns:
            The inverse transformed dataset of the same type as the input data.
        """
        raise NotImplementedError()



class MaxAbsScaler(Scaler[StoreType]):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """    
    def __init__(self) -> None:
        self.scale = None
    
    def fit(self, data:StoreType):
        if isinstance(data, np.ndarray):
            self.scale = np.max(np.abs(data))
        elif isinstance(data, Tensor):
            self.scale = data.abs().max(axis=0).values
        else:
            self.scale = np.max(np.abs(data))
        

    def transform(self, data)-> StoreType:
        # (b , n)  or (n) 
        return data/self.scale 
  
    def inverse_transform(self, data:StoreType) -> StoreType:
        return data*self.scale 
        
    # def __call__(self, tensor:Tensor):
    #     for ch in tensor:
    #         scale = 1.0 / (ch.max(dim=0)[0] - ch.min(dim=0)[0])        
    #         ch.mul_(scale).sub_(ch.min(dim=0)[0])        
    #     return tensor


# class RowMaxAbsScaler(object):
#     """
#     shape of data :  (N , n)
#     - N : sample num
#     - n : node num
#     Transforms each channel to the range [0, 1].
#     """    
#     def __init__(self, device='cpu') -> None:
#         self.scale = None
#         self.device = device
    
#     def fit(self, data:Tensor):
#         size = data.shape
#         self.scale= torch.ones(size[1]).to(self.device )
#         self.scale = data.abs().max(axis=1).values

#     def transform(self, data:Tensor):
#         # (b , n)  or (n) 
#         return data/self.scale 
  
#     def inverse_transform(self, data:Tensor):
#         return data*self.scale 


class MinMaxScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """    
    def __init__(self, device='cpu') -> None:
        self.scale = None
        self.device = device
    
    def fit(self, data:StoreType):
        size = data.shape
        self.scale= torch.ones(size[1]).to(self.device )
        self.scale = data.max(axis=0).values

    def transform(self, data:Tensor):
        # (b , n)  or (n) 
        return data/self.scale 
  
    def inverse_transform(self, data:Tensor):
        return data*self.scale 



class StandarScaler(Scaler):
    """
    shape of data :  (N , n)
    - N : sample num
    - n : node num
    Transforms each channel to the range [0, 1].
    """    
    def __init__(self, device='cpu') -> None:
        self.scale = None
        self.device = device
    
    def fit(self, data:Tensor):
        size = data.shape
        self.scale= torch.ones(size[1]).to(self.device )
        self.scale = data.max(axis=0).values

    def transform(self, data:Tensor):
        # (b , n)  or (n) 
        return data/self.scale 
  
    def inverse_transform(self, data:Tensor):
        return data*self.scale 
