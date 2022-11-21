import math
from turtle import forward
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_timeseries.datasets.electricity import Electricity
from torch_timeseries.nn.dialted_inception import DilatedInception
from torch_timeseries.nn.timeseries_startconv import TimeSeriesStartConv
import torch.nn.functional as F
import torch
from sklearn.preprocessing import StandardScaler
from torch.optim import Optimizer , Adam
from torch.nn import init
import numbers
import torch.optim as optim

from torch_timeseries.data.scaler import MaxAbsScaler



class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    """Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # layer_norm_affline 会增大 参数个数。
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
    def __init__(self, input_node:int,seq_len:int,in_dim:int, embed_dim:int, middle_channel=32, seq_out:int=1,dilation_exponential=1, layers=3, dropout=0.3) -> None:
        super().__init__()
        
        self.input_node = input_node
        self.embed_dim = embed_dim
        
        self.dropout = dropout
        
        self.start_conv = TimeSeriesStartConv(channel_in=in_dim, channel_out=middle_channel)
        self.filter_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.dil_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
    

        
        self.seq_out = seq_out
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
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=middle_channel,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=middle_channel,
                                             out_channels=self.seq_out,
                                             kernel_size=(1,1),
                                             bias=True)
        # self.end_conv = nn.Conv2d(in_channels=middle_channel,
        #                                      out_channels=seq_out,
        #                                      kernel_size=(1,1),
        #                                      bias=True)
        # self.device = device
        
        self.idx = torch.arange(self.input_node)

        kernel_size = 7
        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)
                # self.dil_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.filter_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(middle_channel, middle_channel, dilation_factor=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=middle_channel,
                                                out_channels=middle_channel,
                                                kernel_size=(1, 1)))

                layer_norm_affline = False
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
        
        for i in range(self.layers):
            residual = x
            # x = self.dil_convs[i](x)
            # x = F.dropout(x, self.dropout, training=self.training) # (n , 1 , m , p)
            # x = F.sigmoid(x)
        
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate # (n , 1 , m , p)
            x = F.dropout(x, self.dropout, training=self.training) # (n , 1 , m , p)
            
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

            idx = None
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)
                
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
        
        
        
        
    
    
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))



if __name__ == "__main__":
    import sys
    import numpy as np
    import os
    sys.path.insert(0, os.path.abspath('/notebooks/pytorch_timeseries'))
    from torch_timeseries.datasets import ExchangeRate
    
    window = 168
    horizon = 3
    electricity = Electricity(root='./data', window=window, horizon=horizon)
    # exchange_rate = ExchangeRate(root='./data', window=window, horizon=horizon)
    
    
    dataset = electricity
    device = 'cuda:0'
    scaler = MaxAbsScaler(device=device)
    scaler.fit(dataset.raw_tensor)
    
    
    import torch
    from torch.utils.data.dataset import random_split, Subset
    
    from torch.utils.data import RandomSampler
    

    dataset_size = len(dataset)
    # seed_generator = torch.Generator()
    # seed_generator.manual_seed(42)
    train_size = int(0.6 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - test_size - train_size
    
    # fixed suquence dataset
    indices = range(0, len(dataset))
    train_dataset = Subset(dataset, indices[0 : train_size ])
    val_dataset = Subset(dataset, indices[train_size :(test_size+ train_size) ])
    test_dataset = Subset(dataset, indices[ -val_size  :])
    assert len(train_dataset) + len(test_dataset) + len(val_dataset) == dataset_size
    
    # train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size,val_size], generator=seed_generator)
    
    
    
    # torch.manual_seed(10)
    # torch.cuda.manual_seed(10)
    
    # 为了使得DataLoader 每次随机变化我们需要指定 RamdomSampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=RandomSampler(train_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,sampler=RandomSampler(val_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32,sampler=RandomSampler(test_dataset))
    
    
    

    model = Net(input_node=dataset.num_nodes,seq_len=dataset.window,in_dim=1, embed_dim=8, middle_channel=8, seq_out=1,dilation_exponential=2, layers=5)
    model = model.to(device)
    # loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    # for x, y in loader:
    
    nParams = sum([p.nelement() for p in model.parameters()])
    print([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    loss_func = torch.nn.L1Loss(size_average=False)
    best_val = 10000000
    
    weight_decay = 0.00001
    optim = Optim(
        model.parameters(),'adam', 0.001, 5, lr_decay=0.00001
    )
    # optim = Adam(
    #     model.parameters(),  lr=0.001, weight_decay=weight_decay)



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
            import pdb
            pdb.set_trace()         
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
        test_rse =  normal_std( normed_test_tensor)
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
            
            output:Tensor = model(x) # ( b, seq_out_len, n, 1)
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
    for epoch in range(50):
        train(train_loader=train_loader, model=model, optim=optim)
        
        rse, rae, correlation = evaluate(val_loader=val_loader, model=model)
        print('| end of epoch {:3d}  | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, rse, rae, correlation))
        
            
            
            
        

  
        # print(n(x).shape) # torch.Size([64, 16, 321, 48])
        
        
        

        




