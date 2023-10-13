import torch
from torch_geometric_temporal.nn.recurrent import DCRNN as DCRNNLayer

from torch_timeseries.utils.adj import adj_to_edge_index_weight


class DCRNN(torch.nn.Module):
    def __init__(self, seq_len, out_len,predefined_A,K=2):
        super(DCRNN, self).__init__()
        self.layer = DCRNNLayer(seq_len, out_len, K)
        # self.lin_x = torch.nn.Linear(hidden_size, input_size)
        self.edge_index, self.edge_weight = adj_to_edge_index_weight(predefined_A)
        
        self.edge_index = torch.tensor(self.edge_index)
        self.edge_weight = torch.tensor(self.edge_weight)


    def forward(self, x):
        # x: [batch_size,num_nodes, seq_len]
        ys = []
        for i in range(x.shape[0]):  
            import pdb;pdb.set_trace()      
            yi = self.layer(x[i],self.edge_index,self.edge_weight)
            ys.append(yi)
        y = torch.stack(ys,dim=0)
        return y

