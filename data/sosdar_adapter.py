import mmcv
import numpy as np
import os
import cv2
import torch
import open3d as o3d
from mmdet.datasets import DATASETS, PIPELINES
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.points import LiDARPoints
from mmcv.parallel import DataContainer as DC
from mmcv.parallel import DataContainer as DC

# ====================================================================
# [1] 工具函数
# ====================================================================
def parse_osdar23_calibration(calib_file, target_camera='rgb_center'):
    if not os.path.exists(calib_file):
        return np.eye(4)
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    intrinsics = np.eye(3)
    extrinsics = np.eye(4)
    found_sensor = False
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line.startswith("data_folder:"):
            folder_name = line.split(":")[1].strip()
            found_sensor = (folder_name == target_camera)
        if found_sensor:
            if line.startswith("camera_matrix:"):
                matrix_str = line.split("[")[1]
                while "]" not in matrix_str:
                    matrix_str += lines[idx].strip()
                    idx += 1
                matrix_str = matrix_str.replace(']', '').replace(';', ',')
                values = [float(x) for x in matrix_str.split(',') if x.strip()]
                if len(values) == 9: intrinsics = np.array(values).reshape(3, 3)
            if line.startswith("combined homogenous transform:"):
                matrix_str = ""
                while "]" not in matrix_str:
                    matrix_str += lines[idx].strip()
                    idx += 1
                if "[" in matrix_str: matrix_str = matrix_str.split("[")[1]
                matrix_str = matrix_str.replace(']', '').replace(';', ',')
                values = [float(x) for x in matrix_str.split(',') if x.strip()]
                if len(values) == 16: extrinsics = np.array(values).reshape(4, 4)
                break
    
    R_rect = np.array([
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [1,  0,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)
    
    view_pad = np.eye(4)
    view_pad[:3, :3] = intrinsics
    lidar2img = view_pad @ R_rect @ extrinsics
    return lidar2img

# ====================================================================
# [2] Pipeline - FormatPoly (带调试功能)
# ====================================================================
@PIPELINES.register_module()
class FormatPoly(object):
    def __init__(self, bev_size=(224, 512), pc_range=[0, -44.8, -5, 204.8, 44.8, 10]):
        self.bev_size = bev_size # [H, W] = [512, 224] ? No, usually [W, H] in cv2
        # 注意：在 cv2 中，size 是 (width, height)
        # 如果 grid_size 是 [512, 224] (X=512, Y=224)
        # 那么生成的图片应该是 512 高，224 宽
        # np.zeros((H, W)) -> np.zeros((512, 224))
        self.pc_range = pc_range
        self.debug_counter = 0 # 计数器

    def __call__(self, results):
        if 'gt_poly_3d' in results:
            polys = results['gt_poly_3d']
            
            # --- Mask 初始化 ---
            # grid_size=[512, 224], 对应 (X, Y)
            # 图片应该是 (H=512, W=224)
            # 参数 bev_size=(224, 512) 传入的是 (W, H)
            W, H = self.bev_size
            mask = np.zeros((H, W), dtype=np.float32)
            
            x_min, y_min = self.pc_range[0], self.pc_range[1]
            x_max, y_max = self.pc_range[3], self.pc_range[4]
            
            # --- 数据结构清洗 (关键!) ---
            valid_polys = []
            if isinstance(polys, list):
                for p in polys:
                    if isinstance(p, np.ndarray):
                        if p.ndim == 2 and p.shape[0] > 1: # 正常的线 (N, 3)
                            valid_polys.append(p)
                        elif p.ndim == 1 and p.shape[0] > 3: # 可能是被压扁的线
                             valid_polys.append(p.reshape(-1, 3))
            elif isinstance(polys, np.ndarray):
                if polys.ndim == 3: # (M, N, 3)
                    valid_polys = [p for p in polys]
                elif polys.ndim == 2: # (N, 3) 单条线
                    valid_polys = [polys]

            # --- 绘图 ---
            drawn_count = 0
            for poly in valid_polys:
                if len(poly) < 2: continue
                
                pts_img = []
                for pt in poly:
                    # 归一化 0~1
                    # x (前后) -> H (512)
                    # y (左右) -> W (224)
                    nx = (pt[0] - x_min) / (x_max - x_min)
                    ny = (pt[1] - y_min) / (y_max - y_min)
                    
                    # 越界检查
                    if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                        continue

                    # 映射到像素
                    # cv2 坐标是 (x=col, y=row)
                    # 我们希望 y_real 对应 col (Width), x_real 对应 row (Height)
                    # 且 x_real=0 在图像底部? 或者顶部?
                    # 通常 BEV 图下方是车头 (x=0)，上方是远处 (x=max)
                    # 所以 row = (1 - nx) * H  (翻转) 还是 int(nx * H)?
                    # MMDetection3D 默认 coordinate 是 x 轴在前。
                    # 让 x=0 在图像底部 (row=H-1)，x=max 在顶部 (row=0)
                    
                    px = int(ny * W)       # col
                    py = int((1 - nx) * H) # row (翻转X轴，让前方朝上)
                    
                    pts_img.append([px, py])
                
                if len(pts_img) > 1:
                    pts_img = np.array(pts_img, dtype=np.int32)
                    cv2.polylines(mask, [pts_img], isClosed=False, color=1, thickness=3)
                    drawn_count += 1
            
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # [1, H, W]
            results['gt_masks'] = DC(mask_tensor, stack=True)
            
            # --- [DEBUG] 只在前几个样本打印 ---
            self.debug_counter += 1
            if self.debug_counter < 5:
                print(f"\n🔍 [DEBUG FormatPoly] Sample {self.debug_counter}")
                print(f"   Input Polys: {len(polys)} raw elements")
                print(f"   Valid Polys: {len(valid_polys)} lines")
                print(f"   Drawn Lines: {drawn_count}")
                print(f"   Mask Non-Zero: {np.count_nonzero(mask)} pixels")
                if np.count_nonzero(mask) == 0:
                    print("   ⚠️ WARNING: Mask is empty! Check coordinate mapping or poly data.")

        return results

@PIPELINES.register_module()
class LoadSOSDaRPCD(object):
    def __init__(self, load_dim=4, use_dim=4):
        self.load_dim = load_dim
        self.use_dim = use_dim
    def __call__(self, results):
        filename = results['pts_filename']
        try:
            pcd = o3d.io.read_point_cloud(filename)
            points = np.asarray(pcd.points)
            if points.shape[1] == 3:
                points = np.hstack([points, np.zeros((points.shape[0], 1))])
            points = points.astype(np.float32)
            points = LiDARPoints(points, points_dim=points.shape[-1], attribute_dims=None)
            results['points'] = points
            results['pts_fields'] = ['x', 'y', 'z', 'intensity'][:self.use_dim]
        except Exception as e:
            points = LiDARPoints(np.zeros((1, self.use_dim), dtype=np.float32), points_dim=self.use_dim)
            results['points'] = points
        return results

@DATASETS.register_module()
class SOSDaRDataset(Custom3DDataset):
    CLASSES = ('car', 'pedestrian', 'obstacle')

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False):
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=False, test_mode=test_mode)

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            pts_filename=info['lidar_path'],
            sample_idx=info['sample_idx'],
            img_prefix=None,
            img_info=dict(filename=info['img_path']) if info.get('img_path') else None,
            lidar2img=parse_osdar23_calibration(info['calib_path']) if info.get('calib_path') else np.eye(4),
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d
        )
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if annos:
                input_dict['gt_bboxes_3d'] = annos['gt_bboxes_3d']
                input_dict['gt_labels_3d'] = annos['gt_labels_3d']
                # 传递 gt_poly_3d 给 pipeline
                input_dict['gt_poly_3d'] = annos.get('gt_poly_3d', [])
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = info['annos'].get('gt_bboxes_3d', np.zeros((0, 7), dtype=np.float32))
        gt_labels_3d = info['annos'].get('gt_labels_3d', np.zeros((0,), dtype=np.long))
        gt_poly_3d = info['annos'].get('gt_poly_3d', [])

        if len(gt_bboxes_3d) > 0:
            # 补齐速度维度 (7 -> 9)
            if gt_bboxes_3d.shape[1] == 7:
                 gt_bboxes_3d = np.hstack([gt_bboxes_3d, np.zeros((gt_bboxes_3d.shape[0], 2))])
            gt_bboxes_3d_obj = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d_obj = LiDARInstance3DBoxes(np.zeros((0, 9), dtype=np.float32), box_dim=9, origin=(0.5, 0.5, 0.5))

        return dict(
            gt_bboxes_3d=gt_bboxes_3d_obj,
            gt_labels_3d=gt_labels_3d,
            gt_poly_3d=gt_poly_3d
        )