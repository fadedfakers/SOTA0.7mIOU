import torch
import numpy as np
from torch.utils.data import Sampler

class ClassAwareSampler(Sampler):
    def __init__(self, dataset, reduce_factor=1.0):
        self.dataset = dataset
        self.reduce_factor = reduce_factor
        
        # 1. 统计每个类别的样本索引
        # 格式: {class_id: [img_idx1, img_idx2...]}
        self.class_indices = {i: [] for i in range(len(dataset.classes))}
        
        print("Analyzing class distribution for sampler...")
        # 这步可能会慢，建议缓存结果
        for idx in range(len(dataset)):
            # 快速读取 Label (不加载点云)
            ann_info = dataset.get_ann_info(idx) 
            labels = ann_info['gt_labels_3d']
            
            # 如果该帧包含某类，就加入该类的索引列表
            unique_labels = np.unique(labels)
            for label in unique_labels:
                self.class_indices[label].append(idx)
                
        # 2. 计算每个类别的采样数量 (取最大类别的 N% 或者平均值)
        self.class_counts = {k: len(v) for k, v in self.class_indices.items()}
        print(f"Class distribution: {self.class_counts}")
        
        # 简单的平衡策略: 每个类都采样 max_count 个 (Over-sampling)
        self.num_samples_per_class = max(self.class_counts.values())
        
    def __iter__(self):
        # 3. 生成采样列表
        indices = []
        for class_id in self.class_indices:
            class_idxs = self.class_indices[class_id]
            if len(class_idxs) == 0:
                continue
            
            # 补齐到 num_samples_per_class (随机重复采样)
            replace = len(class_idxs) < self.num_samples_per_class
            sampled = np.random.choice(class_idxs, self.num_samples_per_class, replace=replace)
            indices.extend(sampled)
            
        # 打乱顺序
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.class_indices) * self.num_samples_per_class