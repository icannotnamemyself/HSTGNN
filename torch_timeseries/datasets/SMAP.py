import os
from typing import Callable, Optional
from torch_timeseries.data.extract import extract_zip
from .PytDataset import PytDataset
from .utils import download_url






class SMAP(PytDataset):
    
    url = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
    
    resources = {
        'data.zip':'c40a236775c1c3a26de601f66541b414'
    }
    
    
    def __init__(self,root:str="./data",transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.name = "SMAP"
        self.data_raw_folder =os.path.join( root, self.name, "raw")
        self.data_processed_folder =os.path.join( root, self.name, "processed")
        
        path = download_url(self.url, folder=data_raw_folder)
        print(f"extract from {path} to {data_raw_folder}")
        extract_zip(path, data_raw_folder)
        
        
        
        
        

    


