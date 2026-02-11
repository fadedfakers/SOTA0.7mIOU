import torch
import torch.nn as nn

class BEVFusion(nn.Module):
    def __init__(self, lidar_channels=128, camera_channels=128, out_channels=128):
        super().__init__()
        
        # 通道压缩与融合
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(lidar_channels + camera_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制 (SE-Block 变体)，让模型自动学习信任 LiDAR 还是 Camera
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, lidar_bev, camera_bev):
        """
        lidar_bev: [B, C_l, H, W]
        camera_bev: [B, C_c, H, W] (必须已由 ViewTransformer 投影到相同 BEV 空间)
        """
        # 1. 拼接
        cat_feat = torch.cat([lidar_bev, camera_bev], dim=1)
        
        # 2. 卷积融合
        fused_feat = self.reduce_conv(cat_feat)
        
        # 3. 注意力加权 (Re-weighting)
        # 解决不同模态在雨雾天气的置信度问题
        att = self.channel_att(fused_feat)
        out = fused_feat * att
        
        return out