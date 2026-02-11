import argparse
import mmcv
import torch
import numpy as np
import cv2
import os
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.runner import load_checkpoint

# 类别名称映射
CLASS_NAMES = {0: 'Car', 1: 'Ped', 2: 'Obs'}

def project_3d_to_2d(points_3d, lidar2img):
    points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    points_2d_h = (lidar2img @ points_h.T).T
    depth = points_2d_h[:, 2]
    depth_safe = depth.copy()
    depth_safe[np.abs(depth_safe) < 1e-3] = 1e-3
    points_2d = points_2d_h[:, :2] / depth_safe[:, None]
    return points_2d, depth

def draw_bev_mask_on_image(img, bev_mask, lidar2img, pc_range, color=(0, 255, 255), alpha=0.5):
    """
    将 BEV Mask 反投影到图像上
    bev_mask: [H, W] (512, 224)
    pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    if bev_mask is None: return img
    
    # 二值化
    mask = (bev_mask > 0.5).astype(np.uint8)
    if mask.sum() == 0: return img
    
    # 获取 Mask 中所有前景点的坐标
    # 注意: 我们之前的映射是 py=int((1-nx)*H), px=int(ny*W)
    # 现在要反算回去: nx = 1 - py/H, ny = px/W
    
    ys, xs = np.where(mask > 0) # ys是行(H方向), xs是列(W方向)
    if len(ys) == 0: return img
    
    H, W = mask.shape
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    
    # 反归一化
    # nx = 1 - ys / H  (因为之前翻转了)
    # ny = xs / W
    
    # 稍微稀疏采样一下，防止点太多画太慢
    step = 2 
    ys = ys[::step]
    xs = xs[::step]
    
    nx = 1.0 - (ys / float(H))
    ny = xs / float(W)
    
    real_x = x_min + nx * (x_max - x_min)
    real_y = y_min + ny * (y_max - y_min)
    real_z = np.full_like(real_x, -1.75) # 假设地面高度
    
    points_3d = np.stack([real_x, real_y, real_z], axis=1)
    
    # 投影到图像
    uv, depth = project_3d_to_2d(points_3d, lidar2img)
    uv = uv.astype(np.int32)
    
    # 创建覆盖层
    overlay = img.copy()
    h, w = img.shape[:2]
    
    valid_mask = (depth > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    valid_uv = uv[valid_mask]
    
    # 在覆盖层上画点 (或者画圆)
    for p in valid_uv:
        #cv2.circle(overlay, tuple(p), 1, color, -1)
        overlay[p[1], p[0]] = color # 直接改像素更快
        
        # 简单的膨胀效果
        if p[0]+1 < w: overlay[p[1], p[0]+1] = color
        if p[1]+1 < h: overlay[p[1]+1, p[0]] = color

    # 融合
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def draw_projected_box(img, bboxes_3d, scores, labels, lidar2img, score_thr=0.1):
    if len(bboxes_3d) == 0: return img
    corners_3d = bboxes_3d.corners.numpy()
    
    for i in range(len(corners_3d)):
        score = float(scores[i])
        if score < score_thr: continue
        label_idx = int(labels[i])
        label_name = CLASS_NAMES.get(label_idx, str(label_idx))
        if label_idx == 1: color = (0, 0, 255)   # Ped
        elif label_idx == 0: color = (255, 0, 0) # Car
        else: color = (0, 255, 0)                # Obs

        verts = corners_3d[i]
        uv, depth = project_3d_to_2d(verts, lidar2img)
        uv = uv.astype(np.int32)
        if np.sum(depth > 0.5) < 2: continue

        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        try:
            for p1, p2 in lines:
                if depth[p1] > 0.5 and depth[p2] > 0.5:
                    cv2.line(img, tuple(uv[p1]), tuple(uv[p2]), color, 2)
            # Label
            if depth[0] > 0.5:
                cv2.putText(img, f"{label_name} {score:.2f}", tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except: pass
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default='vis_results', help='Output directory')
    parser.add_argument('--thr', default=0.25, type=float, help='Visualization threshold')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1, shuffle=False, dist=False)

    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    print(f"Starting BEV Visualization (Thr={args.thr})...")
    pc_range = cfg.point_cloud_range
    
    # 预先定义一个上采样函数 (28x64 -> 512x224)
    # 其实模型 forward 里如果已经做了 interpolate 就不用了，
    # 但为了保险，我们这里检查一下
    
    for i, data in enumerate(data_loader):
        if i >= 20: break 
        
        img_metas = data['img_metas'][0].data[0][0]
        if 'img_info' in img_metas: img_path = img_metas['img_info']['filename']
        elif 'filename' in img_metas: img_path = img_metas['filename']
        else: continue
            
        lidar2img = img_metas['lidar2img'] 
        if not os.path.exists(img_path): continue
        raw_img = cv2.imread(img_path)
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        
        boxes_3d = result.get('boxes_3d', [])
        scores_3d = result.get('scores_3d', [])
        labels_3d = result.get('labels_3d', [])
        
        # [核心] 获取 BEV Mask
        bev_mask = result.get('bev_seg_mask', None)
        
        # 1. 画轨道 (黄色半透明)
        if bev_mask is not None:
            # 确保尺寸是 (512, 224)
            if bev_mask.shape != (512, 224):
                bev_mask = cv2.resize(bev_mask, (224, 512), interpolation=cv2.INTER_LINEAR)
            
            raw_img = draw_bev_mask_on_image(raw_img, bev_mask, lidar2img, pc_range, color=(0, 255, 255))

        # 2. 画检测框
        if len(boxes_3d) > 0:
            raw_img = draw_projected_box(raw_img, boxes_3d, scores_3d, labels_3d, lidar2img, score_thr=args.thr)

        out_name = os.path.join(args.out_dir, f'vis_{os.path.basename(img_path)}')
        cv2.imwrite(out_name, raw_img)
        print(f"Saved {out_name}")

if __name__ == '__main__':
    main()