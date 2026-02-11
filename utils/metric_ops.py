import torch
import numpy as np
import cv2

# ============================
# PyTorch Version (For Loss)
# ============================

def chamfer_distance_torch(source, target):
    """
    计算倒角距离 (Batch支持)
    source: [B, N, 3] (Pred)
    target: [B, M, 3] (GT)
    """
    # 扩展维度以便广播计算: [B, N, M, 3]
    # source[:, :, None, :] - target[:, None, :, :] 
    # 计算 L2 距离平方
    dists = torch.cdist(source, target, p=2) # [B, N, M]
    
    # min_dist_source_to_target: 每个预测点离最近GT点的距离
    min_dists1, _ = torch.min(dists, dim=2) # [B, N]
    term1 = torch.mean(min_dists1, dim=1)   # [B]
    
    # min_dist_target_to_source: 每个GT点离最近预测点的距离
    min_dists2, _ = torch.min(dists, dim=1) # [B, M]
    term2 = torch.mean(min_dists2, dim=1)   # [B]
    
    return term1 + term2

# ============================
# Numpy Version (For Evaluation)
# ============================

def chamfer_distance_numpy(source, target):
    """
    单样本 Numpy 计算
    source: [N, 3]
    target: [M, 3]
    """
    from scipy.spatial import cKDTree
    
    # 使用 KDTree 加速最近邻搜索
    tree_s = cKDTree(source)
    tree_t = cKDTree(target)
    
    # query(x) 返回 (distances, indices)
    dist_s2t, _ = tree_t.query(source, k=1)
    dist_t2s, _ = tree_s.query(target, k=1)
    
    return np.mean(dist_s2t) + np.mean(dist_t2s)

def calc_rail_iou(pred_polys, gt_polys, grid_size=(640, 640), 
                 point_cloud_range=[-50, -50, -5, 50, 50, 3], thickness=5):
    """
    计算轨道 mIoU (通过光栅化)
    Args:
        pred_polys: List of [N, 3] points
        gt_polys: List of [M, 3] points
        grid_size: BEV 网格大小
        thickness: 绘制轨道的线宽 (像素), 模拟一定的容差范围
    """
    canvas_pred = np.zeros(grid_size, dtype=np.uint8)
    canvas_gt = np.zeros(grid_size, dtype=np.uint8)
    
    min_x, min_y = point_cloud_range[0], point_cloud_range[1]
    max_x, max_y = point_cloud_range[3], point_cloud_range[4]
    scale_x = grid_size[0] / (max_x - min_x)
    scale_y = grid_size[1] / (max_y - min_y)
    
    def to_uv(pts):
        uv = np.zeros((pts.shape[0], 2), dtype=np.int32)
        uv[:, 0] = (pts[:, 0] - min_x) * scale_x
        uv[:, 1] = (pts[:, 1] - min_y) * scale_y
        return uv

    # 绘制 GT
    for poly in gt_polys:
        uv = to_uv(poly)
        # cv2.polylines 需要 list of array
        cv2.polylines(canvas_gt, [uv], isClosed=False, color=1, thickness=thickness)
        
    # 绘制 Pred
    for poly in pred_polys:
        uv = to_uv(poly)
        cv2.polylines(canvas_pred, [uv], isClosed=False, color=1, thickness=thickness)
        
    # 计算 IoU
    intersection = np.logical_and(canvas_pred, canvas_gt).sum()
    union = np.logical_or(canvas_pred, canvas_gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return intersection / union