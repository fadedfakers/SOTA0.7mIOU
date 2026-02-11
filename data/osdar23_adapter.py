import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import cv2
import traceback
from mmdet3d.core.bbox import LiDARInstance3DBoxes # [关键引入]
from mmdet3d.datasets.builder import DATASETS, PIPELINES
from mmdet3d.datasets.pipelines import Compose

# ==================================================================
# [1] PCD 加载器 (修复了 NumPy 只读警告)
# ==================================================================
@PIPELINES.register_module()
class LoadSOSDaRPCD(object):
    def __init__(self, load_dim=4, use_dim=4):
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        pts_filename = results['pts_filename']
        try:
            points = self._load_sosdar_pcd(pts_filename).copy()
        except Exception as e:
            return None
            
        if points.shape[1] < self.load_dim:
            N = points.shape[0]
            zeros = np.zeros((N, self.load_dim - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, zeros])
            
        from mmdet3d.core.points import LiDARPoints
        results['points'] = LiDARPoints(points, points_dim=points.shape[1])
        return results

    def _load_sosdar_pcd(self, filepath):
        with open(filepath, 'rb') as f:
            num_points = 0
            while True:
                line = f.readline()
                if not line: break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str.startswith('POINTS'):
                    num_points = int(line_str.split()[-1])
                if line_str.startswith('DATA'):
                    break
            buffer = f.read()
            if num_points == 0: return np.zeros((0, 3), dtype=np.float32)
            point_step = len(buffer) // num_points
            raw_data = np.frombuffer(buffer, dtype=np.uint8).reshape(num_points, point_step)
            xyz = np.frombuffer(raw_data[:, :12].tobytes(), dtype=np.float32).reshape(-1, 3)
            return xyz

# ==================================================================
# [2] 数据集类 (修复了 BBox 包装问题)
# ==================================================================
@DATASETS.register_module()
class OSDaR23Dataset(Dataset):
    def __init__(self, data_root, ann_file, pipeline, classes=None, test_mode=False, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(ann_file)
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        self.pipeline = Compose(pipeline)
        
        self.bev_range = [-50, -50, 50, 50]
        self.bev_res = 0.2
        self.line_width = 3

    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            return pickle.load(f)

    def _prepare_data(self, index):
        info = self.data_infos[index]
        full_lidar_path = os.path.join(self.data_root, info['lidar_path'])
        
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=full_lidar_path,
            lidar_path=full_lidar_path,
            img_prefix=None, sweeps=[], timestamp=0,
            img_shape=(800, 1333, 3),
            ori_shape=(800, 1333, 3),
            pad_shape=(800, 1333, 3),
            scale_factor=1.0, 
            img_fields=[],
            bbox3d_fields=[],
            box_type_3d='LiDAR'
        )
        
        annos = info.get('annos', {})
        gt_bboxes_3d = annos.get('gt_bboxes_3d', None)
        
        # [核心修复] 使用 LiDARInstance3DBoxes 包装 NumPy 数组
        if gt_bboxes_3d is not None and len(gt_bboxes_3d) > 0:
            # 确保输入是 torch.Tensor
            boxes_tensor = torch.tensor(gt_bboxes_3d, dtype=torch.float32)
            input_dict['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                boxes_tensor, 
                box_dim=7, 
                origin=(0.5, 0.5, 0.5)
            )
            input_dict['gt_labels_3d'] = annos.get('gt_labels_3d', None)
            input_dict['bbox3d_fields'].append('gt_bboxes_3d')
        else:
            # 如果没有框，也要给一个空的 Box 对象，防止转换失败
            input_dict['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                torch.zeros((0, 7)), box_dim=7
            )
        
        input_dict['gt_poly_3d'] = annos.get('gt_poly_3d', [])

        try:
            return self.pipeline(input_dict)
        except Exception as e:
            # 调试时非常有用
            # print(f"❌ Pipeline 报错: {e}")
            return None

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        # 恢复正常的重试逻辑
        while True:
            data = self._prepare_data(idx)
            if data is None:
                idx = np.random.randint(0, len(self.data_infos))
                continue
            return data

    # ==========================================================
    # [3] BEV mIoU 评估接口
    # ==========================================================
    def evaluate(self, results, metric='mIoU', **kwargs):
        print("\n" + "="*50)
        print("🚀 执行阶段一：轨道几何 mIoU 评估...")
        ious = []
        canvas_h = int((self.bev_range[3] - self.bev_range[1]) / self.bev_res)
        canvas_w = int((self.bev_range[2] - self.bev_range[0]) / self.bev_res)

        for i in range(len(results)):
            info = self.data_infos[i]
            gt_polys = info.get('annos', {}).get('gt_poly_3d', [])
            # 检查输出 Key 是否为 pts_rail
            pred_polys = results[i].get('pts_rail', [])

            if not gt_polys: continue

            gt_mask = self._rasterize(gt_polys, canvas_h, canvas_w)
            pred_mask = self._rasterize(pred_polys, canvas_h, canvas_w)

            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            ious.append(inter / (union + 1e-6))

        miou = np.mean(ious) if ious else 0
        print(f"✅ 评估完成 | 预期目标 > 65% | 当前结果 mIoU: {miou:.4f}")
        print("="*50 + "\n")
        return {'mIoU': miou}

    def _rasterize(self, polys, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            if isinstance(poly, torch.Tensor): poly = poly.cpu().numpy()
            pts = poly[:, :2]
            pixel_pts = (pts - [self.bev_range[0], self.bev_range[1]]) / self.bev_res
            pixel_pts = pixel_pts.astype(np.int32)
            for j in range(len(pixel_pts) - 1):
                cv2.line(mask, tuple(pixel_pts[j]), tuple(pixel_pts[j+1]), 1, self.line_width)
        return mask