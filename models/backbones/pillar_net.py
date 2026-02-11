import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES

@BACKBONES.register_module()
class PillarEncoder(BaseModule):
    def __init__(self, 
                 voxel_size=[0.16, 0.16, 4], 
                 point_cloud_range=[-50, -50, -5, 50, 50, 3], 
                 in_channels=4, 
                 feat_channels=[64],
                 with_distance=False,
                 voxel_layer=None,
                 frozen_stages=-1,  # [修复] 添加此参数以兼容 Config
                 init_cfg=None,
                 **kwargs):          # [修复] 添加 kwargs 吃掉所有多余参数
        super(PillarEncoder, self).__init__(init_cfg)
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        # 解析网格参数
        self.x_min, self.y_min, self.z_min = point_cloud_range[0:3]
        self.x_max, self.y_max, self.z_max = point_cloud_range[3:6]
        self.vo_x, self.vo_y, self.vo_z = voxel_size
        
        self.grid_w = int((self.x_max - self.x_min) / self.vo_x)
        self.grid_h = int((self.y_max - self.y_min) / self.vo_y)
        
        self.in_channels = in_channels
        self.out_channels = feat_channels[-1]
        
        # 输入维度: x, y, z, i, dt + (x-xc, y-yc, z-zc) -> in_channels + 3
        input_dim = in_channels + 3
        
        self.pfn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points):
        """
        points: List[Tensor], 每个 Tensor [N, 5] (x, y, z, i, dt)
        """
        device = points[0].device
        dtype = points[0].dtype
        
        processed_batches = []
        
        for pts in points:
            if pts.shape[0] > 0:
                # 1. 计算索引
                x_idx = ((pts[:, 0] - self.x_min) / self.vo_x).long()
                y_idx = ((pts[:, 1] - self.y_min) / self.vo_y).long()
                
                # 2. 过滤越界点
                mask = (x_idx >= 0) & (x_idx < self.grid_w) & \
                       (y_idx >= 0) & (y_idx < self.grid_h)
                
                pts_valid = pts[mask]
                x_idx = x_idx[mask]
                y_idx = y_idx[mask]
                
                if pts_valid.shape[0] == 0:
                     processed_batches.append(
                        torch.zeros((self.out_channels, self.grid_h, self.grid_w), 
                                    device=device, dtype=dtype))
                     continue

                # 3. 增强点特征
                x_center = x_idx * self.vo_x + self.x_min + self.vo_x / 2
                y_center = y_idx * self.vo_y + self.y_min + self.vo_y / 2
                z_center = pts_valid[:, 2]
                
                pts_aug = torch.cat([
                    pts_valid, 
                    pts_valid[:, 0:1] - x_center.unsqueeze(1),
                    pts_valid[:, 1:2] - y_center.unsqueeze(1),
                    pts_valid[:, 2:3] - z_center.unsqueeze(1)
                ], dim=1)
                
                # 4. 特征提取
                feat = self.pfn(pts_aug)
                
                # 5. Scatter Max
                bev_map = self._scatter_max(feat, x_idx, y_idx, device, dtype)
                processed_batches.append(bev_map)
                
            else:
                processed_batches.append(
                    torch.zeros((self.out_channels, self.grid_h, self.grid_w), 
                                device=device, dtype=dtype)
                )
        
        return torch.stack(processed_batches)

    def _scatter_max(self, feat, x_idx, y_idx, device, dtype):
        indices = y_idx * self.grid_w + x_idx
        C = feat.shape[1]
        canvas = torch.zeros((C, self.grid_h * self.grid_w), device=device, dtype=dtype)
        
        # 简单 Scatter 实现 (覆盖模式)
        # 如果追求更高性能，可以使用 torch_scatter
        canvas[:, indices] = feat.t()
            
        return canvas.view(C, self.grid_h, self.grid_w)