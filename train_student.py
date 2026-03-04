import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import GraphDataset
from modeling.vectornet import HGNN
from modeling.student_vectornet import LightVectorNet 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16 
    
    # 训练超参数
    epochs = 50      # 测试阶段使用 50 轮，正式训练建议调至 200 轮
    lr = 0.002       # 初始学习率
    alpha = 0.5      # Loss 融合权重: 0.5 * Hard_Loss(Ground Truth) + 0.5 * Soft_Loss(Teacher)

    print("[INFO] 开始加载训练数据集...")
    train_dataset = GraphDataset('./interm_data/train_intermediate') 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("[INFO] 初始化 Teacher 模型 (HGNN, in_channels=8)...")
    teacher = HGNN(in_channels=8, out_channels=60, num_subgraph_layers=3).to(device)
    teacher_weights_path = './pretrained_teacher/epoch_24.valminade_2.637.200624.xkhuang.pth' 
    checkpoint = torch.load(teacher_weights_path, map_location=device)
    teacher.load_state_dict(checkpoint['state_dict']) 
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("[INFO] 初始化 Student 模型 (LightVectorNet, in_channels=8)...")
    student = LightVectorNet(
        in_channels=8, 
        out_channels=60,
        num_subgraph_layers=2,   
        subgraph_width=64,       
        global_graph_width=128,  
        traj_pred_mlp_width=256  
    ).to(device)
    
    save_path = './student_light_vectornet.pth'
    if os.path.exists(save_path):
        print(f"[INFO] 检测到本地权重文件 {save_path}，正在加载预训练参数...")
        student.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("[INFO] 未检测到本地权重文件，模型将随机初始化进行训练。")

    student.train()
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 损失函数配置
    criterion_hard = nn.SmoothL1Loss(reduction='none') 
    criterion_soft = nn.SmoothL1Loss(reduction='none')

    print(f"[INFO] 开始知识蒸馏训练 | Alpha: {alpha} | Initial LR: {lr} | Epochs: {epochs}")
    print("-" * 60)
    for epoch in range(epochs):
        accum_loss, accum_hard, accum_soft = 0.0, 0.0, 0.0
        for step, data in enumerate(train_loader):
            data = data.to(device)
            y_true = data.y.view(-1, 60)
            
            # 数据预处理：对齐至 8 维输入特征
            data_for_model = data.clone()
            actual_dim = data.x.size(1) 
            if actual_dim == 4:
                padding = torch.zeros(data.x.size(0), 4).to(device)
                data_for_model.x = torch.cat([data.x, padding], dim=1)
            elif actual_dim > 8:
                data_for_model.x = data.x[:, :8]

            # 前向传播
            with torch.no_grad():
                y_teacher = teacher(data_for_model)
            y_student = student(data_for_model)
            
            # 损失计算与动态权重分配
            loss_hard_base = criterion_hard(y_student, y_true).mean(dim=1) 
            loss_soft_base = criterion_soft(y_student, y_teacher).mean(dim=1) 
            movement = torch.abs(y_true).sum(dim=1) 
            weight = torch.where(movement > 2.0, 10.0, 1.0).to(device)
            
            loss_hard = (loss_hard_base * weight).mean()
            loss_soft = (loss_soft_base * weight).mean()
            loss = alpha * loss_hard + (1.0 - alpha) * loss_soft
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()
            
            accum_loss += loss.item()
            accum_hard += loss_hard.item()
            accum_soft += loss_soft.item()
            
        avg_loss = accum_loss / len(train_loader)
        sample_pred = y_student[0, :5].detach().cpu().numpy()
        
        # 终端日志输出
        print(f"[Epoch {epoch+1:03d}/{epochs:03d}] Loss_Total: {avg_loss:.4f} | Loss_Soft: {accum_soft/len(train_loader):.4f}")
        torch.save(student.state_dict(), './student_light_vectornet.pth')
        print(f"    -> [DEBUG] 样本 0 预测输出序列(前5维): {sample_pred}")
        
        scheduler.step()

    print("[INFO] 训练流程结束，模型权重已保存。")

if __name__ == '__main__':
    main()