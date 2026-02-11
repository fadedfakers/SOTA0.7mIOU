# 文件路径: utils/losses.py

import torch
import torch.nn as nn

class ChamferDistanceLoss(nn.Module):
    """
    [v2.0 新增] 倒角距离损失 (Chamfer Distance Loss)
    计算预测点集 (Predicted Poly3D) 与 真值点集 (GT Poly3D) 之间的双向最近邻距离。
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ChamferDistanceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred_points, gt_points):
        """
        Args:
            pred_points: (B, N, 3) 预测的轨道采样点 (通常 N=10~20)
            gt_points: (B, M, 3) 真值轨道点 (M 可以不等于 N)
        Returns:
            loss: 标量
        """
        # 维度检查
        assert pred_points.dim() == 3 and gt_points.dim() == 3, \
            f"Expected (B, N, 3), got {pred_points.shape} and {gt_points.shape}"

        # 1. 计算成对距离矩阵 (B, N, M)
        # dist[b, i, j] = ||pred[b, i] - gt[b, j]||
        # 使用 L2 范数 (欧氏距离)
        dist_matrix = torch.cdist(pred_points, gt_points, p=2)
        
        # 2. 对于每个预测点，找最近的 GT 点 (Pred -> GT)
        # min_dist_pred: (B, N)
        min_dist_pred, _ = torch.min(dist_matrix, dim=2)
        
        # 3. 对于每个 GT 点，找最近的预测点 (GT -> Pred)
        # min_dist_gt: (B, M)
        min_dist_gt, _ = torch.min(dist_matrix, dim=1)
        
        # 4. 计算平均距离 (双向)
        loss_pred = torch.mean(min_dist_pred, dim=1) # (B,)
        loss_gt = torch.mean(min_dist_gt, dim=1)     # (B,)
        
        loss = loss_pred + loss_gt
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return self.loss_weight * loss

def build_loss(cfg):
    """简易的 Loss 构建工厂"""
    if cfg['type'] == 'ChamferDistanceLoss':
        return ChamferDistanceLoss(
            loss_weight=cfg.get('loss_weight', 1.0),
            reduction=cfg.get('reduction', 'mean')
        )
    elif cfg['type'] == 'L1Loss':
        return nn.L1Loss(reduction='mean')
    elif cfg['type'] == 'MSELoss':
        return nn.MSELoss(reduction='mean')
    else:
        raise NotImplementedError(f"Loss type {cfg['type']} not implemented")