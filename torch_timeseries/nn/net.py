import math
from turtle import forward
from torch import Tensor
import torch.nn as nn
from torch_timeseries.datasets.electricity import DataLoader
from torch_timeseries.nn.dialted_inception import DilatedInception
from torch_timeseries.nn.timeseries_startconv import TimeSeriesStartConv
import torch.nn.functional as F
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import Optimizer , Adam
from torch.nn import init
import numbers

from torch_timeseries.data.scaler import MaxAbsScaler




class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    """Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class Net(nn.Module):
    def __init__(self, input_node:int,seq_len:int,in_dim:int, embed_dim:int, middle_channel=32, seq_out:int=1,dilation_exponential=1, layers=3, dropout=0.3, device='cpu') -> None:
        super().__init__()
        
        self.input_node = input_node
        self.embed_dim = embed_dim
        
        self.dropout = dropout
        
        self.start_conv = TimeSeriesStartConv(channel_in=in_dim, channel_out=middle_channel)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.dil_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
    

        
        self.seq_len = seq_len # 输入窗口长度 window
        self.layers = layers

        # 数据长度应该大于=感受野，否则kernelsize 需要发生变化
        max_kernel_size = 7
        if dilation_exponential>1:
            self.max_receptive_field = int(1+(max_kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1)) 
        else:
            self.max_receptive_field = layers*(max_kernel_size-1) + 1
        # self.end_conv_2 = nn.Conv2d(in_channels=embed_dim,
        #                                      out_channels=seq_out,
        #                                      kernel_size=(1,1),
        #                                  self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
        self.end_conv = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=seq_out,
                                             kernel_size=(1,1),
                                             bias=True)
        self.device = device
        
        self.idx = torch.arange(self.input_node).to(device)


        for i in range(1):
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential>1:
                    rf_size_i = int(1 + i*(max_kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
                else:
                    rf_size_i = i*layers*(max_kernel_size-1)+1
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (max_kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(max_kernel_size-1)
                # self.dil_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.filter_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))

                layer_norm_affline = True
                if self.seq_len>self.max_receptive_field:
                    self.norm.append(LayerNorm((middle_channel, input_node, self.seq_len - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((middle_channel, input_node, self.max_receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential
    def forward(self,input:Tensor, idx=None):
        """_summary_
        Args:
            x (Tensor): shape (b ,aux_dim, n , p)
            b: batch_size
            aux_dim : normally 1
            n: node num
            p: 窗口时间维度
        """
        if self.seq_len<self.max_receptive_field:
            input = nn.functional.pad(input,(self.max_receptive_field-self.seq_len,0,0,0))
        
        
        x = self.start_conv(input)  # out: (b, embed_dim, n, p )
        # x = F.relu(x)
        x = F.tanh(x)
        # x3 = self.dil(x2)   # out: (b, embed_dim, n, p-6)
        # x4 = F.sigmoid(x3)
        # x5 = self.end_conv_1(x4)
        # x6 = self.end_conv_2(x5)
        
        for i in range(self.layers):
            # x = self.dil_convs[i](x)
            # x = F.dropout(x, self.dropout, training=self.training) # (n , 1 , m , p)
            # x = F.sigmoid(x)
        
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate # (n , 1 , m , p)
            x = F.dropout(x, self.dropout, training=self.training) # (n , 1 , m , p)
            
            

            idx = None
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        x = self.end_conv(x)
        return x
        
        
        
        
    
    



if __name__ == "__main__":
    import sys
    import numpy as np
    import os
    sys.path.insert(0, os.path.abspath('/notebooks/pytorch_timeseries'))
    from torch_timeseries.datasets import Electricity
    
    window = 60
    horizon = 1
    electricity = Electricity(root='./data', window=window, horizon=horizon)
    
    
    device = 'cuda:0'
    scaler = MaxAbsScaler(device=device)
    scaler.fit(electricity.raw_tensor)
    
    
    import torch
    from torch.utils.data.dataset import random_split

    dataset_size = len(electricity)
    seed_generator = torch.Generator()
    seed_generator.manual_seed(42)
    train_size = int(0.6 * len(electricity))
    test_size = int(0.2 * len(electricity))
    val_size = len(electricity) - test_size - train_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(electricity, [train_size, test_size,val_size], generator=seed_generator)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    
    model = Net(input_node=electricity.num_nodes,seq_len=electricity.window,in_dim=1, embed_dim=8, middle_channel=8, seq_out=1,dilation_exponential=2, layers=4)
    model = model.to(device)
    # loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    # for x, y in loader:
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    loss_func = torch.nn.L1Loss()
    best_val = 10000000
    
    weight_decay = 0.00001
    optim = Adam(
        model.parameters(),  lr=0.001, weight_decay=weight_decay)



    def evaluate(val_loader: DataLoader, model, evaluateL2):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        for X, Y in val_loader:
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


            evaluateL2 = nn.MSELoss(size_average=False).to(device)
            evaluateL1 = nn.L1Loss(size_average=False).to(device)
            total_loss += evaluateL2( scaler.inverse_transform(output), scaler.inverse_transform(Y)).item()
            total_loss_l1 += evaluateL1(scaler.inverse_transform(output), scaler.inverse_transform(Y)).item()
            
            n_samples += (output.size(0) * electricity.num_nodes)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae

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
        
        model.train()
        for train_x, train_y in train_loader:
            model.zero_grad()

            train_x = train_x.to(device, dtype=torch.float)            
            train_y = train_y.to(device, dtype=torch.float)            
            # x = torch.zeros_like(train_x)
            total_loss = 0
            x = scaler.transform(train_x)
            
            # for i in x:
            #     x[i] = scaler.transform(train_x) 
            x = x.unsqueeze(1) 
            x = x.transpose(2,3 )   # torch.Size([64, 1, 321, 60])
            
            output:Tensor = model(x) # ( b, seq_out_len, n, 1)
            output = output.squeeze(dim=3) # -> ( b, seq_out_len, n)
            output = output.transpose(1,2)  # -> ( b, n, seq_out_len)
            
            output = output.squeeze(dim=2) # -> ( b,  n)
            
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
    for i in range(50):
        train(train_loader=train_loader, model=model, optim=optim)
            
            
            
        

  
        # print(n(x).shape) # torch.Size([64, 16, 321, 48])
        
        
        

        




