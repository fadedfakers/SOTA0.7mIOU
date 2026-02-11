import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS, build_loss

@HEADS.register_module()
class BEVSegHead(nn.Module):
    def __init__(self, 
                 in_channels=384, 
                 num_classes=1, 
                 loss_seg=dict(type='DiceLoss', loss_weight=1.0),
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.loss_seg_func = build_loss(loss_seg)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        # x shape: [B, C, H_feat, W_feat]
        return self.conv(x)

    def loss(self, preds, gt_masks):
        losses = dict()
        if gt_masks is None:
            losses['loss_seg'] = preds.sum() * 0
            return losses
            
        # 确保 gt_masks 维度正确 [B, 1, H, W]
        if gt_masks.dim() == 3:
            target = gt_masks.unsqueeze(1).float()
        else:
            target = gt_masks.float()
        
        # [核心修复] 对齐尺寸
        # preds: [B, 1, 28, 64]
        # target: [B, 1, 512, 224]
        # 我们把 preds 上采样到 target 的大小
        
        if preds.shape[-2:] != target.shape[-2:]:
            preds = F.interpolate(preds, size=target.shape[-2:], mode='bilinear', align_corners=False)
            
        losses['loss_seg'] = self.loss_seg_func(preds, target)
        return losses
    
    def get_seg_masks(self, preds, img_metas=None):
        # 推理时也可能需要上采样，或者直接返回低分辨率mask让后处理去搞
        # 为了方便可视化，这里直接上采样
        # 假设标准尺寸是 512x224 (需要根据 config 调整，或者动态获取)
        # 这里简单起见，先不上采样，或者上采样到一个固定值
        target_size = (512, 224) 
        if preds.shape[-2:] != target_size:
             preds = F.interpolate(preds, size=target_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(preds)