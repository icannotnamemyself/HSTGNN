import pytest
import torch
from torch_timeseries.nn.attention import FullAttention, ProbAttention



def test_fullattention():
    
    atten = FullAttention(mask_flag=True)
    batch_size = 64
    seq_len = 128
    num_heads = 8
    d_model = 512
    
    queries = torch.rand(batch_size, seq_len, num_heads, d_model)
    keys = torch.rand(batch_size, seq_len, num_heads, d_model)
    values = torch.rand(batch_size, seq_len, num_heads, d_model)
    
    output = atten(queries, keys, values)
    
    assert output[0].shape == (batch_size, seq_len, num_heads, d_model)



def test_probattention():
    
    atten = ProbAttention(mask_flag=True)
    batch_size = 64
    seq_len = 128
    num_heads = 8
    d_model = 512
    
    queries = torch.rand(batch_size, seq_len, num_heads, d_model)
    keys = torch.rand(batch_size, seq_len, num_heads, d_model)
    values = torch.rand(batch_size, seq_len, num_heads, d_model)
    
    output = atten(queries, keys, values)
    
    assert output[0].shape == (batch_size, seq_len, num_heads, d_model)
