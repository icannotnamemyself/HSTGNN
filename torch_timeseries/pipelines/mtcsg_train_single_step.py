import math
import sys
import numpy as np
import os

from torch_timeseries.datasets.electricity import Electricity
from torch_timeseries.nn.net import Net, Optim
from torch_timeseries.nn.mtcsg import MTCSG
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


def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
    return torch.matrix_power(x, d)



def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1
    
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))



def train_pipe(dataset:Dataset , device:str, scaler:Scaler, model:nn.Module, loss_func:nn.Module, optim:Optimizer,epoches:int, batch_size=64, lambda_A=0,c_A=1, weight_threshold=0.1):
    """traning pipe line for ((n, p, m)) -> (n, )

    Args:
        dataset (Dataset): _description_
        window (int): _description_
        horizon (int): _description_
        device (str): _description_
        scaler (Scaler): _description_
        model (nn.Module): _description_
        loss_func (nn.Module): _description_
        optim (Optimizer): _description_
        epoches (int): _description_

    Returns:
        _type_: _description_
    """
    # fit scaler
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
    sampler_seed = 42 
    sampler_generator = torch.Generator()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset, generator=sampler_generator))
    val_loader = DataLoader(val_dataset, batch_size=batch_size,sampler=RandomSampler(val_dataset,generator=sampler_generator))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,sampler=RandomSampler(test_dataset,generator=sampler_generator))
    
    
    model = model.to(device=torch.device(device))
    
    
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
        # normed_test_tensor = scaler.transform(test_raw_tensor)
        test_rse =  normal_std(test_raw_tensor)
        test_rae = torch.mean(torch.abs(test_raw_tensor - torch.mean(test_raw_tensor)))
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
        
        # 让 train_loader 每次都random不一样，能够显著的提升模型性能。
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

            
            # lagrange term
            
            h_A = _h_A(model.weighted_A, dataset.feature_nums)
            loss = loss_func(predict, ground_truth)
            
            loss +=  lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(model.weighted_A*model.weighted_A)
            
            
            loss.backward()
            
            total_loss += loss.item()
            grad_norm = optim.step()
            
            
            model.weighted_A.data.data = stau(model.weighted_A.data, weight_threshold)
            
            if iter%100==0:
                print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0))/output.size(1)))
            iter += 1
            
            
    for epoch in range(epoches):
        train(train_loader=train_loader, model=model, optim=optim)
        rse, rae, correlation = evaluate(val_loader=val_loader, model=model)
        print('| end of epoch {:3d}  | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, rse, rae, correlation))
        
            
            
if __name__ == "__main__":
    device = 'cuda:1'
    window = 168
    horizon = 3
    epoches = 50
    dataset = Electricity(root='./data', window=window, horizon=horizon)
    model = MTCSG(node_num=dataset.feature_nums,seq_len=dataset.window,in_dim=1, middle_channel=64, seq_out=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer=Optim(model.parameters(),'adam', 0.001, 5, lr_decay=0.00001)
    scaler = MaxAbsScaler(device)
    
    c_A = 1
    lambda_A = 0
    
    total_runs = 3
    k_max_iter = 3
    h_A_new = torch.tensor(1.)
    
    # for i in range(total_runs):
    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            
            train_pipe(dataset, device, scaler, model,  torch.nn.L1Loss(size_average=False), optimizer, epoches, c_A =c_A ,lambda_A=lambda_A)
            
            # update parameters
            A_new = model.weighted_A.data.clone()
            h_A_new = _h_A(A_new, dataset.feature_nums)
            if h_A_new.item() > 0.25 * model.h_A:
                c_A*=10
            else:
                break
        
        model.h_A = h_A_new.item()
        lambda_A += c_A * h_A_new.item()
