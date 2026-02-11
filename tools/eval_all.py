import argparse
import mmcv
import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

# ====================================================================
# [1] 辅助函数：渲染 BEV Mask
# ====================================================================
def render_rail_bev_from_polys(polys, pc_range, canvas_size=(512, 224), thickness=3):
    """从 PolyLine (真值) 渲染 Mask"""
    # 注意: canvas_size 应该是 (H, W) = (512, 224) 还是 (W, H)?
    # 根据之前的 Adapter 代码，grid_size=[512, 224] 对应 (X, Y)
    # 所以 H=512 (X轴), W=224 (Y轴)
    # cv2 也是 (W, H)
    
    H, W = canvas_size[0], canvas_size[1]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    valid_polys = []
    if isinstance(polys, list):
        for p in polys:
            if isinstance(p, np.ndarray) and p.ndim == 2: valid_polys.append(p)
    elif isinstance(polys, np.ndarray) and polys.ndim == 3:
        valid_polys = [p for p in polys]
        
    for poly in valid_polys:
        if len(poly) < 2: continue
        pts_img = []
        for pt in poly:
            nx = (pt[0] - x_min) / (x_max - x_min)
            ny = (pt[1] - y_min) / (y_max - y_min)
            
            # 映射像素
            px = int(ny * W)       # col
            py = int((1 - nx) * H) # row (翻转)
            pts_img.append([px, py])
            
        if len(pts_img) > 1:
            pts_img = np.array(pts_img, dtype=np.int32)
            cv2.polylines(mask, [pts_img], isClosed=False, color=1, thickness=thickness)
    return mask

def process_pred_mask(pred_mask, target_size=(512, 224)):
    """处理预测 Mask: 确保尺寸正确 + 二值化"""
    if pred_mask is None:
        return np.zeros(target_size, dtype=np.uint8)
        
    # 如果预测尺寸不对，插值
    if pred_mask.shape != target_size:
        pred_mask = cv2.resize(pred_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        
    # 二值化 (阈值 0.5)
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    return binary_mask

# ====================================================================
# [2] 主逻辑
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Detection & Segmentation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+', default=['mAP'], help='eval metrics')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    
    # 强制开启 Test 模式但保留真值读取
    cfg.data.test.test_mode = False 
    
    print("🔍 [Step 1] 初始化数据集...")
    dataset = build_dataset(cfg.data.test)
    
    # [Patch] 修复真值字段以支持 mAP 评估
    print("🛠️ [FIX] Patching dataset annotations...")
    for i in range(len(dataset.data_infos)):
        info = dataset.data_infos[i]
        if 'annos' not in info: info['annos'] = {}
        annos = info['annos']
        gt_bboxes = annos.get('gt_bboxes_3d', [])
        annos['gt_num'] = len(gt_bboxes)
        if isinstance(gt_bboxes, np.ndarray):
            if gt_bboxes.shape[1] > 7: gt_bboxes = gt_bboxes[:, :7]
        annos['gt_boxes_upright_depth'] = gt_bboxes
        if 'gt_labels_3d' in annos: annos['class'] = annos['gt_labels_3d']
        else: annos['class'] = np.zeros(0, dtype=int)

    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    
    print("🔍 [Step 2] 构建模型...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # 统计变量
    seg_tp, seg_fp, seg_fn = 0, 0, 0
    results_list = []
    
    # 获取 BEV 尺寸参数
    pc_range = cfg.point_cloud_range
    # grid_size [X, Y, Z] -> H=X, W=Y
    bev_h, bev_w = cfg.grid_size[0], cfg.grid_size[1] 
    canvas_size = (bev_h, bev_w) # (512, 224)

    print(f"🚀 [Step 3] 开始推理与评估 (共 {len(dataset)} 帧)...")
    
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        
        # --- A. 收集检测结果 ---
        # 裁剪 9维 -> 7维
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            if boxes.tensor.shape[1] == 9:
                boxes.tensor = boxes.tensor[:, :7]
                boxes.box_dim = 7
        results_list.append(result)
        
        # --- B. 计算分割 IoU ---
        # 1. 获取真值 Poly
        gt_info = dataset.get_ann_info(i)
        gt_polys = gt_info.get('gt_poly_3d', [])
        if hasattr(gt_polys, 'data'): gt_polys = gt_polys.data
            
        # 2. 渲染真值 Mask
        mask_gt = render_rail_bev_from_polys(gt_polys, pc_range, canvas_size, thickness=5)
        
        # 3. 获取预测 Mask (BEV Seg)
        mask_pred_raw = result.get('bev_seg_mask', None)
        mask_pred = process_pred_mask(mask_pred_raw, target_size=canvas_size)
        
        # 4. 统计混淆矩阵
        intersection = np.logical_and(mask_pred, mask_gt).sum()
        union = np.logical_or(mask_pred, mask_gt).sum()
        
        seg_tp += intersection
        seg_fp += (mask_pred.sum() - intersection)
        seg_fn += (mask_gt.sum() - intersection)

    # ===================================================
    # [Step 4] 输出报告
    # ===================================================
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION REPORT")
    print("="*50)
    
    # 1. Detection mAP
    print("\n[1] Object Detection Results:")
    try:
        dataset.evaluate(results_list, metric=args.eval)
    except Exception as e:
        print(f"⚠️ Detection evaluation failed: {e}")

    # 2. Segmentation mIoU
    rail_iou = seg_tp / (seg_tp + seg_fp + seg_fn + 1e-6)
    print("\n[2] Rail Segmentation Results:")
    print(f"   +----------------+---------+")
    print(f"   | Metric         | Value   |")
    print(f"   +----------------+---------+")
    print(f"   | Rail BEV mIoU  | {rail_iou:.4f}  |")
    print(f"   +----------------+---------+")
    print("="*50)

if __name__ == '__main__':
    main()