import torch
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import GraphDataset
from modeling.vectornet import HGNN
from modeling.student_vectornet import LightVectorNet

def calculate_metrics(pred_traj, gt_traj):
    """
    计算 ADE (Average Displacement Error) 和 FDE (Final Displacement Error)
    输入 Tensor 形状: [batch_size, num_steps, 2]
    """
    distances = torch.norm(pred_traj - gt_traj, dim=-1) 
    ade = distances.mean(dim=1).mean().item()
    fde = distances[:, -1].mean().item()
    return ade, fde

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16 

    print("[INFO] 开始加载测试数据集...")
    test_dataset = GraphDataset('./interm_data/test_intermediate') 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("[INFO] 加载 Teacher 模型权重 (HGNN)...")
    teacher = HGNN(in_channels=8, out_channels=60, num_subgraph_layers=3).to(device)
    teacher.load_state_dict(torch.load('./pretrained_teacher/epoch_24.valminade_2.637.200624.xkhuang.pth', map_location=device)['state_dict'])
    teacher.eval()

    print("[INFO] 加载 Student 模型权重 (LightVectorNet)...")
    student = LightVectorNet(
        in_channels=8, out_channels=60, num_subgraph_layers=2, 
        subgraph_width=64, global_graph_width=128, traj_pred_mlp_width=256
    ).to(device)
    student.load_state_dict(torch.load('./student_light_vectornet.pth', map_location=device))
    student.eval()

    total_teacher_ade, total_teacher_fde = 0.0, 0.0
    total_student_ade, total_student_fde = 0.0, 0.0
    num_batches = 0

    print("[INFO] 启动全量数据的 ADE/FDE 量化计算流程...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            y_true = data.y.view(-1, 30, 2).cumsum(dim=1) 
            
            # 数据预处理：对齐至 8 维输入特征
            data_for_model = data.clone()
            actual_dim = data.x.size(1) 
            if actual_dim == 4:
                padding = torch.zeros(data.x.size(0), 4).to(device)
                data_for_model.x = torch.cat([data.x, padding], dim=1)
            elif actual_dim > 8:
                data_for_model.x = data.x[:, :8]

            # 模型预测与坐标还原
            pred_teacher = teacher(data_for_model).view(-1, 30, 2).cumsum(dim=1)
            pred_student = student(data_for_model).view(-1, 30, 2).cumsum(dim=1)
            
            # 计算当前 Batch 误差
            t_ade, t_fde = calculate_metrics(pred_teacher, y_true)
            s_ade, s_fde = calculate_metrics(pred_student, y_true)
            
            total_teacher_ade += t_ade
            total_teacher_fde += t_fde
            total_student_ade += s_ade
            total_student_fde += s_fde
            num_batches += 1

    print("\n" + "=" * 50)
    print("                实验评估报告 (Evaluation Metrics)")
    print("=" * 50)
    print("模型名称\t\t\tADE (m)\t\tFDE (m)")
    print("-" * 50)
    print(f"Teacher (HGNN)\t\t\t{total_teacher_ade/num_batches:.4f}\t\t{total_teacher_fde/num_batches:.4f}")
    print(f"Student (LightVectorNet)\t{total_student_ade/num_batches:.4f}\t\t{total_student_fde/num_batches:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()