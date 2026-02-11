import numpy as np
import torch

def poly_to_points(coeffs, x_range=(-50, 50), num_points=10, z_val=-1.8):
    """
    将多项式系数转换为离散控制点
    Args:
        coeffs (list/array): 多项式系数 [a, b, c, d] -> y = ax^3 + bx^2 + cx + d
        x_range (tuple): x轴采样范围
        num_points (int): 采样点数
        z_val (float): 轨道的地面高度 (LiDAR 坐标系通常为负值)
    Returns:
        points (np.ndarray): [num_points, 3] (x, y, z)
    """
    xs = np.linspace(x_range[0], x_range[1], num_points)
    # numpy.polyval 需要系数最高次幂在前 [a, b, c, d]
    ys = np.polyval(coeffs, xs)
    zs = np.full_like(xs, z_val)
    
    return np.stack([xs, ys, zs], axis=1)

def rotate_points_2d(points, angle):
    """
    二维平面旋转 (用于 transforms.py 中的增强)
    Args:
        points (np.ndarray): [N, 3] or [N, 2]
        angle (float): 弧度制
    """
    # 构建旋转矩阵 (逆时针)
    rot_cos = np.cos(angle)
    rot_sin = np.sin(angle)
    rot_mat = np.array([
        [rot_cos, -rot_sin],
        [rot_sin,  rot_cos]
    ])
    
    # 仅旋转 x, y
    points_xy = points[:, :2]
    points_rot_xy = points_xy @ rot_mat.T
    
    if points.shape[1] == 3:
        return np.hstack([points_rot_xy, points[:, 2:3]])
    return points_rot_xy

def fit_poly_from_points(points, order=3):
    """
    [后处理] 将预测的点集拟合回多项式系数
    Args:
        points (np.ndarray): [N, 3]
        order (int): 多项式阶数 (通常为3: ax^3+...)
    Returns:
        coeffs (np.ndarray): [order+1]
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # 处理垂直线问题：如果 x 变化极小（车辆横向开），多项式拟合会失败
    # v2.0 报告建议直接使用点集输出，但为了兼容旧接口，这里加个保护
    if np.max(x) - np.min(x) < 1.0:
        # 退化为直线 x = c (无法用 y=f(x) 表示)
        # 这里返回全0或其他标志位
        return np.zeros(order + 1)

    try:
        coeffs = np.polyfit(x, y, order)
    except np.linalg.LinAlgError:
        coeffs = np.zeros(order + 1)
        
    return coeffs

def sample_bezier_curve(control_points, num_samples=50):
    """
    (可选) 如果 v2.0 后期改用贝塞尔曲线，可用此函数生成平滑轨迹
    """
    pass # 暂保留接口