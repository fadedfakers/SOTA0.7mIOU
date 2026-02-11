import torch
import torch.nn as nn
from mmdet.models.builder import NECKS

@NECKS.register_module()
class TemporalFusion(nn.Module):
    """
    [Phase 2 更新] 多模态融合模块 (Global Context Fusion)
    替代了原先仅支持单模态时序的 ConvGRU 版本。
    """
    def __init__(self, 
                 in_channels=384,      # SECONDFPN 输出 (128*3)
                 img_channels=256,     # Image FPN 输出
                 out_channels=384,     # 融合后输出
                 frames_num=1, 
                 fusion_method='mvx'):
        super().__init__()
        self.frames_num = frames_num
        self.fusion_method = fusion_method
        
        # 1. 图像特征压缩层
        self.img_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # [B, C, 1, 1]
            nn.Conv2d(img_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 2. 融合卷积层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels + 64, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, pts_feats, img_feats, img_metas):
        # 1. 取出 LiDAR BEV 特征
        if isinstance(pts_feats, (list, tuple)):
            bev_feat = pts_feats[0]
        else:
            bev_feat = pts_feats
            
        # 2. 如果没有图像特征 (兼容性)
        if img_feats is None:
            return [bev_feat]

        # 3. 取 FPN 最后一层
        img_feat = img_feats[-1] 
        
        # 4. 全局上下文融合
        img_global = self.img_mlp(img_feat) 
        
        # 处理 Batch
        B_bev = bev_feat.shape[0]
        B_img = img_global.shape[0]
        
        if B_img != B_bev:
            num_views = B_img // B_bev
            img_global = img_global.view(B_bev, num_views, 64, 1, 1).mean(dim=1)
            
        # 广播
        img_global = img_global.expand(-1, -1, bev_feat.shape[2], bev_feat.shape[3])
        
        # 5. 拼接与融合
        fused_feat = torch.cat([bev_feat, img_global], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        return [fused_feat]