import torch
import torch.nn as nn
from modeling.selfatten import SelfAttentionLayer
from modeling.subgraph import SubGraph
from torch_geometric.utils import to_dense_batch

# 🚀 植入防弹版 MLP (不会脑死亡)
class RobustMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),  # 👈 核心：LeakyReLU 允许负梯度回传，永不坏死
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class LightVectorNet(nn.Module):
    
    def __init__(self, in_channels=8, out_channels=60, num_subgraph_layers=2, 
                 subgraph_width=64, global_graph_width=128, traj_pred_mlp_width=256):
        
        super(LightVectorNet, self).__init__()
        
        # 🚀 核心修改：回归真实的物理公式！
        # 因为 SubGraph 每层翻倍，所以输出维度 = 输入通道 * 2 的层数次方
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers) 
        
        # 1. 局部特征提取
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        
        # 2. 全局注意力层 (现在它会根据 32 维还是 64 维自动对齐了)
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width, need_scale=False)
        
        # 3. 终极预测 MLP
        # 注意：这里 RobustMLP 的输入维度会自动变成 32+128=160 (如果 layers=2)
        # 或者 64+128=192 (如果 layers=3)
        self.traj_pred_mlp = RobustMLP(
            self.polyline_vec_shape + global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, data):
        # 🚀 防御性编程：克隆数据，防止被老师模型干扰
        data_copy = data.clone()
        
        sub_graph_out = self.subgraph(data_copy)
        
        # 将平摊的数据重新按 Batch 分组
        true_batch_size = data_copy.num_graphs if hasattr(data_copy, 'num_graphs') else sub_graph_out.batch.max().item() + 1
        x, mask = to_dense_batch(sub_graph_out.x, sub_graph_out.batch, batch_size=true_batch_size)
        
        valid_lens = mask.sum(dim=1) 
        # 此时 x 的维度是 64，self_atten_layer 也能处理 64，完美匹配！
        out = self.self_atten_layer(x, valid_lens)
        
        # 特征融合并预测
        fused_agent_feature = torch.cat([x[:, 0], out[:, 0]], dim=-1)
        pred = self.traj_pred_mlp(fused_agent_feature)
        return pred