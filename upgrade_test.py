import torch
from torch_geometric.data import Data
import os

def upgrade_pyg_data(file_path):
    if not os.path.exists(file_path):
        print(f"[错误] 找不到文件: {file_path} (请确认你已经解压了数据)")
        return
        
    print(f">>> 正在读取旧版缓存: {file_path}")
    # 1. 强行加载旧版 PyG 序列化的元组 (Data, slices)
    old_data, slices = torch.load(file_path)
    
    # 2. 实例化一个当前新版 PyG 支持的干净 Data 对象
    new_data = Data()
    
    # 3. 核心黑客科技：绕过类的限制，直接去底层的 __dict__ 里把张量偷出来，塞进新对象
    for key, value in old_data.__dict__.items():
        # 过滤掉旧版残留的内部隐藏属性
        if not key.startswith('_'):
            new_data[key] = value
            
    # 4. 用升级后的对象覆盖掉原来的旧文件
    torch.save((new_data, slices), file_path)
    print(f"[成功] 升级完毕！已完美兼容当前环境: {file_path}\n")

if __name__ == "__main__":
    # 🚀 专门升级测试集 (Test Set)
    upgrade_pyg_data('./interm_data/test_intermediate/processed/dataset.pt')