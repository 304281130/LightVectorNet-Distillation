import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(X, valid_len):
    """
    针对 Batch 数据的三维 Masked Softmax (防 NaN 增强版)
    """
    if valid_len is None:
        return F.softmax(X, dim=-1)
    
    bs, n, _ = X.shape 
    # 构造掩码 [Batch, 1, Nodes]
    mask = torch.arange(n, device=X.device)[None, :] < valid_len[:, None]
    
    # 【核心修复 1】：用 -1e9 替代 float('-inf')，防止 -inf 减去 -inf 产生 NaN
    X = X.masked_fill(~mask[:, None, :], -1e9)
    
    # 计算 Softmax
    attention_weights = F.softmax(X, dim=-1)
    
    # 【核心修复 2】：防止全 0 样本（valid_len=0）导致整个权重依然存在异常
    # 把完全无效的样本强制设为 0
    attention_weights = attention_weights.masked_fill(~mask[:, None, :], 0.0)
    
    return attention_weights

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionLayer, self).__init__()
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.need_scale = need_scale

    def forward(self, x, valid_len):
        bs, n, dim = x.shape
        
        query = self.q_lin(x) 
        key = self.k_lin(x)
        value = self.v_lin(x)

        # 核心：bmm 会严格检查第一维是否等于 batch_size (16)
        scores = torch.bmm(query, key.transpose(1, 2)) 
        
        # 兼容性修复：如果 teacher 没有 need_scale 属性，手动给它一个
        use_scale = getattr(self, 'need_scale', False)
        if use_scale:
            scores = scores / (query.size(-1) ** 0.5)
        
        attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(attention_weights, value)