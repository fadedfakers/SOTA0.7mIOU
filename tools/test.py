import argparse
import mmcv
import os
import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='Debug Test Script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--eval', type=str, nargs='+', help='eval metrics')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    print("🔍 [DEBUG] 正在初始化数据集...")
    dataset = build_dataset(cfg.data.test)
    
    # ---------------------------------------------------------
    # [暴力修复 1] 补全真值字段 (gt_num, gt_boxes_upright_depth)
    # ---------------------------------------------------------
    print("🛠️ [FIX] Patching dataset annotations for evaluation...")
    for i in range(len(dataset.data_infos)):
        info = dataset.data_infos[i]
        if 'annos' not in info: info['annos'] = {}
        annos = info['annos']
        
        gt_bboxes = annos.get('gt_bboxes_3d', [])
        annos['gt_num'] = len(gt_bboxes)
        
        # 确保是 numpy 且只取前7维 (防止真值也是9维导致报错)
        if isinstance(gt_bboxes, np.ndarray):
            if gt_bboxes.shape[1] > 7:
                gt_bboxes = gt_bboxes[:, :7]
        annos['gt_boxes_upright_depth'] = gt_bboxes
        
        if 'gt_labels_3d' in annos:
            annos['class'] = annos['gt_labels_3d']
        else:
            annos['class'] = np.zeros(0, dtype=int)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False
    )
    
    print("🔍 [DEBUG] 正在构建模型并加载权重...")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    
    print("🚀 [DEBUG] 开始推理循环...")
    outputs = single_gpu_test(model, data_loader)

    # ---------------------------------------------------------
    # [暴力修复 2] 裁剪预测框 (9维 -> 7维)
    # ---------------------------------------------------------
    print("✂️ [FIX] Truncating predicted boxes to 7 dimensions...")
    for result in outputs:
        if 'boxes_3d' in result:
            boxes = result['boxes_3d']
            # 如果是 9 维 (x,y,z,l,w,h,yaw,vx,vy)，只取前 7 维
            if boxes.tensor.shape[1] == 9:
                boxes.tensor = boxes.tensor[:, :7]
                boxes.box_dim = 7

    print("\n✅ [DEBUG] 推理完成，开始评估...")
    if args.eval:
        dataset.evaluate(outputs, metric=args.eval)

if __name__ == '__main__':
    main()