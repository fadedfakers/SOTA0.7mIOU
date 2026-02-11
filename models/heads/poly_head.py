import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
import numpy as np

class ChamferDistanceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ChamferDistanceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, source, target):
        # source: [B, N, 3], target: [B, M, 3]
        # 计算两组点云之间的倒角距离
        dists1 = torch.cdist(source, target) # [B, N, M]
        
        min_dists1, _ = torch.min(dists1, dim=2) 
        term1 = torch.mean(min_dists1, dim=1) 
        
        min_dists2, _ = torch.min(dists1, dim=1)
        term2 = torch.mean(min_dists2, dim=1) 
        
        return self.loss_weight * (torch.mean(term1) + torch.mean(term2))

@HEADS.register_module()
class PolyHead(BaseModule): # 注意类名要与 Config 一致，如果是 RailPolyHead 请修改这里
    def __init__(self, 
                 in_channels=128, 
                 num_polys=2,
                 num_control_points=20, 
                 hidden_dim=256,
                 loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0),
                 pc_range=[-50, -50, -5, 50, 50, 3], 
                 init_cfg=None):
        super(PolyHead, self).__init__(init_cfg)
        
        self.num_polys = num_polys
        self.num_points = num_control_points
        self.pc_range = pc_range
        
        # 全局特征聚合 + MLP 回归
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_polys * num_control_points * 3) # 输出所有点的坐标
        )
        
        if loss_poly['type'] == 'ChamferDistanceLoss':
            self.loss_func = ChamferDistanceLoss(loss_weight=loss_poly['loss_weight'])
        else:
            self.loss_func = nn.MSELoss()

    def forward(self, x):
        B = x.shape[0]
        feat = self.global_pool(x).view(B, -1)
        
        # 1. 回归 [B, num_polys*num_points*3]
        raw_pred = self.mlp(feat)
        raw_pred = raw_pred.view(B, self.num_polys * self.num_points, 3)
        
        # 2. Sigmoid 归一化到 [0, 1]
        norm_pred = torch.sigmoid(raw_pred)
        
        # 3. 反归一化到真实物理范围
        x_len = self.pc_range[3] - self.pc_range[0]
        y_len = self.pc_range[4] - self.pc_range[1]
        z_len = self.pc_range[5] - self.pc_range[2]
        
        scale = torch.tensor([x_len, y_len, z_len], device=x.device).view(1, 1, 3)
        offset = torch.tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2]], device=x.device).view(1, 1, 3)
        
        points_pred = norm_pred * scale + offset
        return points_pred

    def loss(self, points_pred, gt_poly_3d):
        loss_dict = dict()
        total_loss = torch.tensor(0.0, device=points_pred.device, requires_grad=True)
        valid_cnt = 0
        
        # 范围过滤
        x_min, y_min, z_min = self.pc_range[0], self.pc_range[1], self.pc_range[2]
        x_max, y_max, z_max = self.pc_range[3], self.pc_range[4], self.pc_range[5]
        
        for i in range(len(points_pred)):
            pred = points_pred[i] # [N_points, 3]
            
            # --- GT 处理 ---
            if gt_poly_3d is None or i >= len(gt_poly_3d): continue
            target = gt_poly_3d[i]
            
            if hasattr(target, 'data'): target = target.data
            
            # 尝试转换为 Tensor
            if isinstance(target, list):
                if len(target) == 0: continue
                try:
                    if torch.is_tensor(target[0]): target = torch.cat(target, dim=0)
                    else: target = torch.tensor(np.concatenate(target, axis=0)).float()
                except: continue
            
            if not torch.is_tensor(target):
                continue
                
            target = target.to(pred.device)
            if target.dim() == 3: target = target.view(-1, 3)
            if target.shape[0] < 2: continue

            # --- 仅保留感知范围内的 GT 点 ---
            mask = (target[:, 0] >= x_min) & (target[:, 0] <= x_max) & \
                   (target[:, 1] >= y_min) & (target[:, 1] <= y_max)
            valid_target = target[mask]
            
            if len(valid_target) < 2: continue
            
            # 计算 Loss
            loss = self.loss_func(pred.unsqueeze(0), valid_target.unsqueeze(0))
            total_loss = total_loss + loss
            valid_cnt += 1
            
        if valid_cnt > 0:
            loss_dict['loss_poly'] = total_loss / valid_cnt
        else:
            loss_dict['loss_poly'] = points_pred.sum() * 0.0 # 保持梯度图完整
            
        return loss_dict