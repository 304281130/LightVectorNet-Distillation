import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import GraphDataset
from modeling.student_vectornet import LightVectorNet
from modeling.vectornet import HGNN 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(">>> 正在加载测试场景...")
    dataset = GraphDataset('./interm_data/test_intermediate') 
    data = dataset[3] # 取第一个场景
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader([data], batch_size=1)
    batch_data = next(iter(loader)).to(device)

    # 1. 初始化并加载高徒 (结构必须与训练一致)
    student = LightVectorNet(
        in_channels=8, subgraph_width=64, global_graph_width=128, traj_pred_mlp_width=256
    ).to(device)
    student.load_state_dict(torch.load('./student_light_vectornet.pth', map_location=device))
    student.eval()
    
    # 2. 初始化并加载名师
    teacher = HGNN(in_channels=8, out_channels=60, num_subgraph_layers=3).to(device)
    teacher_weights_path = './pretrained_teacher/epoch_24.valminade_2.637.200624.xkhuang.pth' 
    checkpoint = torch.load(teacher_weights_path, map_location=device)
    teacher.load_state_dict(checkpoint['state_dict']) 
    teacher.eval()

    # === 🚀 可视化推理时也要对齐 8 维输入 ===
    data_for_model = batch_data.clone()
    actual_dim = batch_data.x.size(1)
    if actual_dim == 4:
        padding = torch.zeros(batch_data.x.size(0), 4).to(device)
        data_for_model.x = torch.cat([batch_data.x, padding], dim=1)
    elif actual_dim > 8:
        data_for_model.x = batch_data.x[:, :8]

    # 3. 师徒同时预测
    print(">>> 师徒正在同时预测...")
    with torch.no_grad():
        student_pred = student(data_for_model) 
        teacher_pred = teacher(data_for_model)
    
    # === 核心修改：动态检测是否存在 Ground Truth ===
    has_gt = batch_data.y is not None and batch_data.y.numel() > 0
    
    if has_gt:
        gt_traj = batch_data.y.view(-1, 60)[0].cpu().numpy().reshape(-1, 2).cumsum(axis=0)
        
    student_traj = student_pred[0].cpu().numpy().reshape(-1, 2).cumsum(axis=0)
    teacher_traj = teacher_pred[0].cpu().numpy().reshape(-1, 2).cumsum(axis=0)
    
    # 4. 绘图
    plt.figure(figsize=(10, 8))
    
    # 如果有真实轨迹，才画出绿线
    if has_gt:
        plt.plot(gt_traj[:, 0], gt_traj[:, 1], color='green', marker='o', label='Ground Truth')
        
    plt.plot(teacher_traj[:, 0], teacher_traj[:, 1], color='blue', marker='s', linestyle='-.', label='Teacher')
    plt.plot(student_traj[:, 0], student_traj[:, 1], color='red', marker='x', linestyle='--', label='Student')
    plt.scatter(0, 0, color='orange', marker='*', s=200, label='Start')
    plt.title("Knowledge Distillation: Fixed Dimension Inference")
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_distillation_result.png', dpi=300)
    print(">>> 图像已保存。由于是 Test 集，您将只看到师徒两人的预测轨迹对比。")

if __name__ == '__main__':
    main()