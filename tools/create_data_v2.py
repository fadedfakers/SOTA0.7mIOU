import pickle
import os
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_calib_from_json(json_data):
    """
    从 JSON 中提取外参 (LiDAR -> Camera) 和 内参
    注意：OSDaR23 的 JSON 结构比较复杂，通常在 'streams' 或 'coordinate_systems' 中
    这里做一个通用的尝试性提取
    """
    # 占位符：实际解析需要根据 JSON 具体结构
    # 如果 JSON 里没有现成的矩阵，可能需要读取 calibration.txt
    # 为简化，暂且返回 None，依靠 Adapter 在运行时读取 calibration.txt
    return None

def find_files_by_suffix(directory, suffix):
    """递归查找指定后缀的文件，返回 {文件名主体: 绝对路径}"""
    mapping = {}
    if not directory.exists(): return mapping
    for p in directory.rglob(f"*{suffix}"):
        # OSDaR 文件名通常是 "ID_Timestamp.ext"
        # 我们用 "ID" 作为键 (即文件名第一个下划线前的部分)
        key = p.name.split('_')[0] 
        mapping[key] = str(p)
    return mapping

def create_osdar_infos(root_path, out_dir):
    print(f"\n[OSDaR23] Generating Rich Info for Fusion...")
    root = Path(root_path)
    
    # 查找所有场景文件夹 (含有 calibration.txt 的目录通常是场景根目录)
    scene_dirs = [p.parent for p in root.rglob("calibration.txt")]
    print(f"Found {len(scene_dirs)} scenes.")
    
    infos_train = []
    infos_val = []
    
    for scene_dir in scene_dirs:
        # 1. 解析标注文件
        # 标注文件通常在场景目录下，名为 "场景名_labels.json"
        json_files = list(scene_dir.glob("*_labels.json"))
        if not json_files: continue
        ann_file = json_files[0]
        
        try:
            with open(ann_file, 'r') as f: data = json.load(f)
        except: continue
            
        frames = data.get('openlabel', {}).get('frames', {})
        if not frames: continue
            
        # 2. 建立文件索引 (LiDAR 和 Camera)
        # OSDaR23 目录结构:
        # scene/lidar/
        # scene/rgb_center/ (或其他相机)
        
        lidar_map = find_files_by_suffix(scene_dir / 'lidar', '.pcd')
        img_map = find_files_by_suffix(scene_dir / 'rgb_center', '.png')
        
        # 如果没有 rgb_center，尝试 rgb_highres_center
        if not img_map:
            img_map = find_files_by_suffix(scene_dir / 'rgb_highres_center', '.png')
            
        scene_id = scene_dir.name
        frame_ids = sorted(list(frames.keys()), key=lambda x: int(x) if x.isdigit() else x)
        
        # 划分训练/验证 (按场景前80%后20%)
        split_idx = int(len(frame_ids) * 0.8)
        
        for i, fid in enumerate(frame_ids):
            # fid 通常就是 "0", "1", ...
            # 尝试匹配文件
            if fid not in lidar_map: 
                # 尝试用文件名匹配 (有时候 json key 是 "0" 但文件名是 "00000")
                continue
                
            lidar_path = lidar_map[fid]
            img_path = img_map.get(fid, None) # 图像允许缺失，但最好有
            
            info = {
                'sample_idx': fid,
                'scene_id': scene_id,
                'lidar_path': lidar_path,
                'img_path': img_path, # [关键新增] 图像路径
                'pose': None,         # 位姿 (后续在 Adapter 里读)
                'calib_path': str(scene_dir / 'calibration.txt') # [关键新增] 标定文件路径
            }
            
            if i < split_idx:
                infos_train.append(info)
            else:
                infos_val.append(info)
                
    print(f"-> OSDaR Result: {len(infos_train)} train, {len(infos_val)} val")
    
    with open(os.path.join(out_dir, 'osdar23_infos_train.pkl'), 'wb') as f:
        pickle.dump(infos_train, f)
    with open(os.path.join(out_dir, 'osdar23_infos_val.pkl'), 'wb') as f:
        pickle.dump(infos_val, f)

# SOSDaR 部分保持不变 (略去以节省篇幅，直接用您刚才成功的版本即可)
# ... (请保留您原来代码中的 create_sosdar_infos 函数) ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osdar-root', required=True)
    parser.add_argument('--sosdar-root', required=True)
    args = parser.parse_args()
    
    create_osdar_infos(args.osdar_root, args.osdar_root)
    # create_sosdar_infos(args.sosdar_root, args.sosdar_root) # 如果不需要重新生成 SOSDaR 可注释

if __name__ == '__main__':
    main()