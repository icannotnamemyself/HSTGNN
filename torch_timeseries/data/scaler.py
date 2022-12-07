import torch
from torch import Tensor


class Scaler(object):
    
        
    def fit(self, data:Tensor):
        pass
    def transform(self, data:Tensor):
        pass
  
    def inverse_transform(self, data:Tensor):
        pass


class MaxAbsScaler(Scaler):
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
        self.scale = data.abs().max(axis=0).values.to(self.device, dtype=torch.float)

    def transform(self, data:Tensor):
        # (b , n)  or (n) 
        return data/self.scale 
  
    def inverse_transform(self, data:Tensor):
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
    
    def fit(self, data:Tensor):
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
