import argparse
import mmcv
import torch
import numpy as np
import os
import cv2
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint
from tqdm import tqdm

def render_rail_bev(polys, pc_range, canvas_size=(512, 512), thickness=2):
    """
    将轨道线点集渲染成 BEV 掩码 (带鲁棒性检查)
    polys: 应该是 List[np.ndarray(N, 3)] 或 np.ndarray(M, N, 3)
    """
    mask = np.zeros(canvas_size, dtype=np.uint8)
    
    # --- [核心修复] 数据结构标准化 ---
    valid_polys = []
    
    # 情况1: 如果是单个 numpy 数组
    if isinstance(polys, np.ndarray):
        if polys.ndim == 3: # (M, N, 3) -> M条线
            valid_polys = [p for p in polys]
        elif polys.ndim == 2: # (N, 3) -> 1条线
            valid_polys = [polys]
            
    # 情况2: 如果是列表 List
    elif isinstance(polys, list):
        if len(polys) == 0:
            pass
        # 检查列表里的元素是什么
        elif isinstance(polys[0], np.ndarray):
            if polys[0].ndim == 2: # List[Array(N,3)] -> 标准格式
                valid_polys = polys
            elif polys[0].ndim == 1: # List[Array(3,)] -> 这其实是一条线
                # 把点列表重新组合成一条线
                valid_polys = [np.array(polys)]
        elif isinstance(polys[0], list): # List[List] -> 可能是点的列表
             valid_polys = [np.array(p) for p in polys]

    # --- 渲染逻辑 ---
    for poly in valid_polys:
        # 过滤无效线
        if not isinstance(poly, np.ndarray) or poly.ndim != 2 or poly.shape[0] < 2:
            continue
            
        pts_2d = []
        for pt in poly:
            # 坐标归一化映射
            # x_idx = (y_real - min_y) / range_y * h
            # y_idx = (x_real - min_x) / range_x * w
            # 注意 OSDaR/Kitti 坐标系: x前, y左
            # BEV 画布:通常 x对应宽(y轴), y对应高(x轴)
            
            # X轴映射 (对应画布高度)
            # 图像坐标系通常左上角是(0,0)，x向下，y向右
            # 这里我们做一个简单的映射: 
            # x_real [-10, 200] -> img_y [H, 0] (翻转，前方在上方)
            # y_real [-40, 40] -> img_x [0, W]
            
            # 使用标准的 min-max 映射
            # pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
            
            # 物理坐标 -> 0~1 归一化
            norm_x = (pt[0] - x_min) / (x_max - x_min)
            norm_y = (pt[1] - y_min) / (y_max - y_min)
            
            # 0~1 -> 像素坐标
            # 将物理世界的 X (前方) 映射为图像的 Y (高度)
            # 将物理世界的 Y (左右) 映射为图像的 X (宽度)
            py = int((1 - norm_x) * canvas_size[0]) # 翻转，让车头朝上
            px = int((1 - norm_y) * canvas_size[1]) # 翻转，配合坐标系习惯
            
            # 简单的越界保护
            px = np.clip(px, 0, canvas_size[1]-1)
            py = np.clip(py, 0, canvas_size[0]-1)
            
            pts_2d.append([px, py])
        
        pts_2d = np.array(pts_2d)
        for i in range(len(pts_2d) - 1):
            cv2.line(mask, tuple(pts_2d[i]), tuple(pts_2d[i+1]), 1, thickness)
            
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    pc_range = cfg.point_cloud_range
    
    print("🔍 Building Model...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    print("🔍 Building Dataset...")
    cfg.data.test.test_mode = False 
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print(f"🚀 Starting Rail mIoU Evaluation on {len(dataset)} samples...")

    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
            
        # 调试：打印第一个样本的预测结构，排查问题
        if i == 0:
            if 'rail_polys' in result:
                rp = result['rail_polys']
                print(f"\n[DEBUG Sample 0] Pred Type: {type(rp)}")
                if isinstance(rp, (list, tuple)):
                    print(f"  Length: {len(rp)}")
                    if len(rp) > 0: print(f"  Element 0 Type: {type(rp[0])}, Shape: {getattr(rp[0], 'shape', 'N/A')}")
                elif hasattr(rp, 'shape'):
                    print(f"  Shape: {rp.shape}")
            else:
                print("\n[DEBUG Sample 0] No 'rail_polys' in result keys:", result.keys())

        if 'rail_polys' not in result: continue
        pred_polys = result['rail_polys']
        
        # 获取真值
        gt_info = dataset.get_ann_info(i)
        gt_polys = gt_info.get('gt_poly_3d', [])
        
        # 如果是 DataContainer，拆包
        if hasattr(gt_polys, 'data'): 
            gt_polys = gt_polys.data

        # 渲染
        canvas_res = (512, 512)
        mask_pred = render_rail_bev(pred_polys, pc_range, canvas_res, thickness=3)
        mask_gt = render_rail_bev(gt_polys, pc_range, canvas_res, thickness=3)

        intersection = np.logical_and(mask_pred, mask_gt).sum()
        union = np.logical_or(mask_pred, mask_gt).sum()

        total_tp += intersection
        total_fp += (mask_pred.sum() - intersection)
        total_fn += (mask_gt.sum() - intersection)

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    
    print("\n" + "="*30)
    print(f"📊 Rail BEV Evaluation Result")
    print(f"   mIoU: {iou:.4f}")
    print("="*30)

if __name__ == '__main__':
    main()