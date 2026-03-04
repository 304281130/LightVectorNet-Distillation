# %%

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
import torch
from utils.feature_utils import compute_feature_for_one_seq, encoding_features, save_features
from utils.config import DATA_DIR, LANE_RADIUS, OBJ_RADIUS, OBS_LEN, INTERMEDIATE_DATA_DIR
from tqdm import tqdm


# %%
def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start
# %%


# 找到这个类定义
class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            # 核心破局点：告诉 PyG，合并数据时把每个图的 cluster 编号依次往后推！
            # 加上单张图的节点上限，比如 71
            return int(self.time_step_len[0])
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return 1
        else:
            return 0

# %%


class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision` 
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def process(self):
        import os
        import glob
        import torch
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        # 假设你的项目中已经定义了 get_fc_edge_index 和 GraphData
        # 如果报错找不到，请确保它们已从对应的 utils 或 core 模块导入

        # 1. 定义 raw 文件夹路径并锁定 .pkl 文件
        raw_path = os.path.join(self.root, 'raw')
        data_path_ls = sorted(glob.glob(os.path.join(raw_path, "*.pkl")))

        print(f"\n>>> [DEBUG] 搜索路径: {os.path.join(raw_path, '*.pkl')}")
        print(f">>> [DEBUG] 成功锁定文件数量: {len(data_path_ls)}")

        if len(data_path_ls) == 0:
            raise ValueError(f"在 {raw_path} 中未找到任何 .pkl 文件！")

        valid_len_ls = []
        data_ls = []

        # 2. 遍历并解析每个 .pkl 文件
        for data_p in tqdm(data_path_ls, desc="正在转换特征"):
            try:
                data = pd.read_pickle(data_p)
                all_in_features = data['POLYLINE_FEATURES'].values[0]
                add_len = data['TARJ_LEN'].values[0]
                
                # --- 重要：每个文件都要重置列表，防止数据累加 ---
                x_ls = []
                edge_index_ls = []
                edge_index_start = 0
                agent_id = 0
                # -------------------------------------------

                # 提取目标值和聚类信息
                y = data['GT'].values[0].reshape(-1).astype(np.float32)
                cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
                valid_len_ls.append(cluster.max())

                # 提取掩码
                traj_mask = data["TRAJ_ID_TO_MASK"].values[0]
                lane_mask = data['LANE_ID_TO_MASK'].values[0]

                # 验证 Agent ID
                assert all_in_features[agent_id][-1] == 0, f"Agent ID 错误: {all_in_features[agent_id][-1]}"

                # 处理轨迹特征 (Trajectories)
                for id_, mask_ in traj_mask.items():
                    data_ = all_in_features[mask_[0]:mask_[1]]
                    # 这里的 get_fc_edge_index 应该是在你的 utils 中定义的函数
                    edge_index_, edge_index_start = get_fc_edge_index(
                        data_.shape[0], start=edge_index_start)
                    x_ls.append(data_)
                    edge_index_ls.append(edge_index_)

                # 处理车道特征 (Lanes)
                for id_, mask_ in lane_mask.items():
                    data_ = all_in_features[mask_[0] + add_len : mask_[1] + add_len]
                    edge_index_, edge_index_start = get_fc_edge_index(
                        data_.shape[0], start=edge_index_start)
                    x_ls.append(data_)
                    edge_index_ls.append(edge_index_)

                # 合并当前文件的特征矩阵和边索引
                edge_index = np.hstack(edge_index_ls)
                x = np.vstack(x_ls)
                data_ls.append([x, y, cluster, edge_index])

            except Exception as e:
                print(f"\n[Error] 处理文件 {data_p} 时出错: {e}")
                continue

        # 3. 全局 Padding 逻辑
        if not valid_len_ls:
            raise ValueError("没有成功处理任何有效数据！")

        padd_to_index = np.max(valid_len_ls)
        feature_len = data_ls[0][0].shape[1]
        g_ls = []

        for ind, tup in enumerate(data_ls):
            # tup 的结构: [0:x, 1:y, 2:cluster, 3:edge_index]
            current_max_cluster = tup[2].max()
            
            # 对特征矩阵 x 进行零填充 (Padding)
            tup[0] = np.vstack([
                tup[0], 
                np.zeros((padd_to_index - current_max_cluster, feature_len), dtype=tup[0].dtype)
            ])
            
            # 对聚类 ID 进行填充
            tup[2] = np.hstack([
                tup[2], 
                np.arange(current_max_cluster + 1, padd_to_index + 1)
            ])

            # 封装为 PyTorch Geometric 的 Data 对象
            g_data = GraphData(
                x=torch.from_numpy(tup[0]),
                y=torch.from_numpy(tup[1]),
                cluster=torch.from_numpy(tup[2]),
                edge_index=torch.from_numpy(tup[3]),
                valid_len=torch.tensor([valid_len_ls[ind]]),
                time_step_len=torch.tensor([padd_to_index + 1])
            )
            g_ls.append(g_data)

        # 4. 保存处理后的数据
        data, slices = self.collate(g_ls)
        torch.save((data, slices), self.processed_paths[0])
        print(f"\n>>> [SUCCESS] 预处理完成！数据已保存至: {self.processed_paths[0]}")


# %%
if __name__ == "__main__":
    for folder in os.listdir(DATA_DIR):
        dataset_input_path = os.path.join(
            INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
        dataset = GraphDataset(dataset_input_path)
        batch_iter = DataLoader(dataset, batch_size=256)
        batch = next(iter(batch_iter))


# %%
