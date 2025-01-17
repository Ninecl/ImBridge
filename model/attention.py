import math
import torch

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(emb_dim, att_dim)
        self.w_k = nn.Linear(emb_dim, att_dim)
        self.w_v = nn.Linear(emb_dim, att_dim)
        self.w_concat = nn.Linear(att_dim, emb_dim)


    def forward(self, x):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out = self.attention(q, k, v)
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        return out


    def split(self, tensor):
        num_nodes, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(num_nodes, self.n_head, d_tensor).transpose(0, 1)
        return tensor


    def concat(self, tensor):
        # input is 3 dimension tensor [num_heads, num_nodes, d_tensor]
        num_heads, num_nodes, d_tensor = tensor.size()
        d_model = num_heads * d_tensor
        tensor = tensor.transpose(0, 1).contiguous().view(num_nodes, d_model)
        return tensor
    
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q, k, v, e=1e-12):
        # input is 3 dimension tensor [num_heads, num_nodes, d_tensor]
        num_heads, num_nodes, d_tensor = k.size()
        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(1, 2)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        # 3. multiply with Value
        z = score @ v
        return z


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AttenEncoderLayer(nn.Module):

    def __init__(self, emb_dim, att_dim, hidden_dim, num_heads, drop_prob):
        super(AttenEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, att_dim, num_heads)
        self.norm1 = LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(emb_dim, hidden_dim, drop_prob=drop_prob)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)


    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(x)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x