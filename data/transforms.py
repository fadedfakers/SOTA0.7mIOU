import numpy as np
import torch
from mmcv.utils import build_from_cfg
from mmdet3d.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RailGlobalRotScaleTrans(object):
    def __init__(self, rot_range, scale_ratio_range, translation_std, update_poly3d=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.update_poly3d = update_poly3d # [v2.0 开关]

    def rotate_poly3d(self, poly_coeffs, angle, x_range=(-50, 50), num_points=10):
        """
        [v2.0 修复逻辑]
        输入: 轨道多项式系数 (y = ax^3 + bx^2 + cx + d)
        输出: 旋转后的控制点 (Points)
        原因: 旋转后的曲线可能不再是 x 的函数 (Vertical Line), 必须用点集表示。
        """
        # 1. 在 x 范围内采样点
        xs = np.linspace(x_range[0], x_range[1], num_points)
        ys = np.polyval(poly_coeffs, xs)
        zs = np.zeros_like(xs) # 假设轨道在地平面，或从系数中获取 z
        
        # 2. 构建旋转矩阵 (逆时针)
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
        
        # 3. 旋转点 (x, y)
        points_xy = np.stack([xs, ys], axis=1)
        points_rot = points_xy @ rot_mat.T
        
        # 返回旋转后的点集 [x', y', z]
        return np.hstack([points_rot, zs[:, None]])

    def __call__(self, input_dict):
        # ... (常规点云/BBox 旋转代码保持不变) ...
        
        # 随机生成旋转角度
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        
        # [v2.0 修复] 同步旋转轨道真值
        if self.update_poly3d and 'gt_poly_3d' in input_dict:
            new_polys = []
            for poly in input_dict['gt_poly_3d']:
                # 假设 poly 是系数 [a, b, c, d]
                # 转换为旋转后的控制点
                rot_points = self.rotate_poly3d(poly, noise_rotation)
                new_polys.append(rot_points)
            
            input_dict['gt_poly_3d'] = new_polys
            # 注意: 如果模型 Head 还是接受系数，这里需要重新 polyfit
            # 但 v2.0 报告建议直接回归 Control Points (Source 344)
            
        return input_dict