from typing import Dict, Tuple
import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader , Subset, RandomSampler, Dataset
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch_timeseries.data.scaler import Scaler
from torch_timeseries.datasets.dataset import TimeSeriesDataset
from torch_timeseries.datasets.splitter import SequenceRandomSplitter, Splitter
from torch_timeseries.datasets.wrapper import MultiStepFlattenWrapper
from torch_timeseries.utils.dataloader import dataloader_to_numpy
# import torch
# # Load the Boston Housing dataset
# boston = load_boston()

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# # Convert the data to XGBoost's DMatrix format
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test)

# # Set the parameters for the XGBoost model
# params = {
#     'objective': 'reg:squarederror',
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     "max_depth":8,
#     "n_jobs":16,
#     'n_estimators': 100,
#     "multi_strategy": 'one_output_per_tree'  , # one_output_per_tree multi_output_tree
# }

# # Train the XGBoost model
# # model = xgb.train(params, dtrain)

# model = xgb.XGBRegressor(**params)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate the mean squared error between the predicted and actual values
# mse = mean_squared_error(y_test, y_pred)

# print(f"Mean Squared Error: {mse:.2f}")





class XGBoostPipeline:
    def __init__(self, model: xgb.XGBRegressor, splitter: SequenceRandomSplitter, steps=1, window=168, horizon=3):
        """
        Initializes an XGBoostPipeline object.

        Args:
            model (xgb.XGBRegressor): The XGBoost regression model to train.
            dataloaders (Dict[str, DataLoader]): A dictionary of PyTorch DataLoader objects
                containing the training and validation datasets.
        """
        self.model = model
        self.splitter = splitter
        self.steps = steps
        self.window = window
        self.horizon = horizon

    def train(self, dataset:TimeSeriesDataset) -> None:
        """
        Trains the XGBoost model using the provided datasets.
        """
        
        # dataset_wrapper = MultiStepFlattenWrapper(dataset, steps=self.steps, window=self.window, horizon=self.horizon)
        train_loader , test_loader , val_loader = self.splitter(dataset)
        
        # Concatenate the batches into a single numpy array
        train_x, train_y = dataloader_to_numpy(train_loader)
        test_x, test_y = dataloader_to_numpy(test_loader)
        val_x, val_y = dataloader_to_numpy(val_loader)
        
        # Train the model
        self.model.fit(train_x, train_y,eval_set=[(val_x, val_y)], verbose=True)

        # get prediction result
        pred_y = self.model.predict(test_x)
        
        # Calculate the mean squared error between the predicted and actual values
        mse = mean_squared_error(pred_y, test_y)
        print(f"Mean Squared Error: {mse:.2f}")

    
    def train1(self):
        train_loader , test_loader , val_loader = self.splitter(dataset)
        for i, data in enumerate(train_loader):
            X_chunk, y_chunk = data[0].numpy(), data[1].numpy()
            dtrain.set_float_info('label', y_chunk)
            dtrain.set_float_info('data', X_chunk.reshape(X_chunk.shape[0], -1))
        
        dtrain.set_float_info('label', y_chunk)
        dtrain.set_float_info('data', X_chunk.reshape(X_chunk.shape[0], -1))

        xgb.train(xgb_model=self.model)
        

    def save_model(self, save_path : str = './save') -> None:
        """_summary_
            dataset: Electricity
            model: XGBoost
                - gpu_id : 0 
                - depth : 5
                ...
                ..
            save_path: 
            
        Args:
            save_path (str, optional): _description_. Defaults to './save'.
        """
        pass




if __name__ == "__main__":
    params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    "max_depth":8,
    "n_jobs":16,
    'n_estimators': 100,
    "multi_strategy": 'one_output_per_tree'  , # one_output_per_tree multi_output_tree
    }

    from torch_timeseries.datasets.electricity import Electricity

    model = xgb.XGBRegressor(**params)
    dataset = MultiStepFlattenWrapper(Electricity(root='./data'), steps=31)
    srs = SequenceRandomSplitter(42, 32)
   
    p = XGBoostPipeline(model, splitter=srs)
    
    p.train(dataset)
    
    