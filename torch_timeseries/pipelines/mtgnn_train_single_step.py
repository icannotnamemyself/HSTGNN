import math
import sys
import numpy as np
import os

from torch_timeseries.datasets.electricity import Electricity
from torch_timeseries.nn.net import Net, Optim
sys.path.insert(0, os.path.abspath('/notebooks/pytorch_timeseries'))
from torch_timeseries.datasets import ExchangeRate
from torch_timeseries.datasets.dataset import Dataset
from torch.utils.data import DataLoader
from torch_timeseries.data.scaler import Scaler
import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split, Subset

from torch.utils.data import RandomSampler
from torch.optim import Optimizer
import torch.optim as optim
from torch_timeseries.data.scaler import MaxAbsScaler



    
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))



def train_pipe(dataset:Dataset , window: int , horizon:int, device:str, scaler:Scaler, model:nn.Module, loss_func:nn.Module, optim:Optimizer,epoches:int):
    scaler.fit(dataset.raw_tensor)
    dataset_size = len(dataset)
    
    
    train_size = int(0.6 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - test_size - train_size
    
    # fixed suquence dataset
    indices = range(0, len(dataset))
    train_dataset = Subset(dataset, indices[0 : train_size ])
    val_dataset = Subset(dataset, indices[train_size :(test_size+ train_size) ])
    test_dataset = Subset(dataset, indices[ -val_size  :])
    assert len(train_dataset) + len(test_dataset) + len(val_dataset) == dataset_size
    

    # data change for every epoches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=RandomSampler(train_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,sampler=RandomSampler(val_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,sampler=RandomSampler(test_dataset))
    
    
    model = model.to(device)
    
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    
    
    
    best_val = 10000000
    
    # weight_decay = 0.00001
    # optim = Optim(
    #     model.parameters(),'adam', 0.001, 5, lr_decay=0.00001
    # )
    

    def evaluate(val_loader: DataLoader, model):
        model.eval()
        test_raw_tensor = torch.zeros(0, dataset.window, dataset.num_nodes).to(device)
        # for i, (x , _) in enumerate(test_loader): 
            

        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        evaluateL2 = nn.MSELoss(size_average=False).to(device)
        evaluateL1 = nn.L1Loss(size_average=False).to(device)

        for X, Y in val_loader:
            X = X.to(device, dtype=torch.float)            
            Y = Y.to(device, dtype=torch.float)            
            test_raw_tensor = torch.cat([test_raw_tensor,X], 0)

            
            X = scaler.transform(X)
            Y = scaler.transform(Y)
            
            X = torch.unsqueeze(X,dim=1)
            X = X.transpose(2,3)
            with torch.no_grad():
                output = model(X)
            output = torch.squeeze(output)
            if len(output.shape)==1:
                output = output.unsqueeze(dim=0)
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))

            

            total_loss += evaluateL2( scaler.inverse_transform(output), scaler.inverse_transform( Y)).item()
            total_loss_l1 += evaluateL1(scaler.inverse_transform(output),  scaler.inverse_transform( Y)).item()
            
            n_samples += (output.size(0) * dataset.num_nodes)

        # for eliminating diference between datasets , divide test_rse and test_rae
        normed_test_tensor = scaler.transform(test_raw_tensor)
        test_rse =  normal_std(normed_test_tensor)
        test_rae = torch.mean(torch.abs(normed_test_tensor - torch.mean(normed_test_tensor)))
        rse = math.sqrt(total_loss / n_samples)   / test_rse
        rae = (total_loss_l1 / n_samples)   / test_rae

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        return rse, rae, correlation

    def train(train_loader:DataLoader, model:nn.Module, optim:Optimizer):
        iter = 0 
        total_loss = 0
        model.train()
        
        # 让train_loader 每次都random不一样，能够显著的提升模型性能。
        for train_x, train_y in train_loader:
            model.zero_grad()
            train_x = train_x.to(device, dtype=torch.float)            
            train_y = train_y.to(device, dtype=torch.float)            
            # x = torch.zeros_like(train_x)
            
            x = scaler.transform(train_x)
            
            # for i in x:
            #     x[i] = scaler.transform(train_x) 
            x = x.unsqueeze(1) 
            x = x.transpose(2,3 )   # torch.Size([64, 1, 321, 60])
            
            output = model(x) # ( b, seq_out_len, n, 1)
            output = torch.squeeze(output) # -> ( b,n)
            
            # output = output.squeeze(dim=3) # -> ( b, seq_out_len, n)
            # output = output.transpose(1,2)  # -> ( b, n, seq_out_len)
            
            # output = output.squeeze(dim=2) # -> ( b,  n)
            
            # inverse scale
            predict = scaler.inverse_transform(output)
            
            ground_truth =  train_y

            loss = loss_func(predict, ground_truth)
            loss.backward()
            
            total_loss += loss.item()
            grad_norm = optim.step()
            
            if iter%100==0:
                print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0))/output.size(1)))
            iter += 1
            
            
    for epoch in range(epoches):
        train(train_loader=train_loader, model=model, optim=optim)
        rse, rae, correlation = evaluate(val_loader=val_loader, model=model)
        print('| end of epoch {:3d}  | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, rse, rae, correlation))
        
            
            
if __name__ == "__main__":
    device = 'cuda:0'
    window = 168
    horizon = 3
    epoches = 50
    dataset = Electricity(root='./data', window=window, horizon=horizon)
    model = Net(input_node=dataset.feature_nums,seq_len=dataset.window,in_dim=1, embed_dim=8, middle_channel=8, seq_out=1,dilation_exponential=2, layers=5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer=Optim(model.parameters(),'adam', 0.001, 5, lr_decay=0.00001)
    scaler = MaxAbsScaler(device)
    
    train_pipe(dataset, window, horizon, device, scaler, model,  torch.nn.L1Loss(size_average=False), optimizer, epoches)