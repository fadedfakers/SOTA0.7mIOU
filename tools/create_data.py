import pickle
import os
import glob
import numpy as np
import argparse
from pathlib import Path
import json
import sys

# ==========================================
# 1. 通用工具
# ==========================================
def build_file_index(root_path):
    print(f"Indexing files in {root_path} ...")
    file_index = {}
    for p in Path(root_path).rglob('*'):
        if p.is_file():
            file_index[p.name] = str(p)
    return file_index

# ==========================================
# 2. OSDaR 逻辑 (保持您已成功的版本)
# ==========================================
def find_osdar_lidar_file(frame_id, json_path_obj, file_index=None):
    fid_str = str(frame_id)
    scene_dir = json_path_obj.parent
    local_lidar_dir = scene_dir / 'lidar'
    
    # 策略1: 局部搜索
    if local_lidar_dir.exists():
        for p in local_lidar_dir.glob("*"):
            if p.name == f"{fid_str}.pcd": return str(p)
            if p.name.startswith(f"{fid_str}_") and p.suffix == '.pcd': return str(p)
    
    # 策略2: 全局搜索
    if file_index:
        candidates = [f"{fid_str}.pcd", f"{fid_str.zfill(5)}.pcd"]
        for cand in candidates:
            if cand in file_index: return file_index[cand]
        prefix = f"{fid_str}_"
        for fname, fpath in file_index.items():
            if fname.startswith(prefix) and fname.endswith('.pcd'):
                if json_path_obj.parent.name in fpath: return fpath
    return None

def create_osdar_infos(root_path, out_dir):
    print(f"\n[OSDaR23] Processing...")
    file_index = build_file_index(root_path)
    all_files = list(Path(root_path).rglob("*.json"))
    valid_ann_files = [p for p in all_files if ("label" in p.name.lower() or "anno" in p.name.lower())]
    
    infos_train = []
    infos_val = []

    for ann_file in valid_ann_files:
        try:
            with open(ann_file, 'r') as f: data = json.load(f)
        except: continue

        frames = None
        if 'openlabel' in data and 'frames' in data['openlabel']: frames = data['openlabel']['frames']
        elif 'frames' in data: frames = data['frames']
        if not frames: continue
            
        scene_id = ann_file.stem.replace('_labels', '').replace('.json', '')
        frame_ids = sorted(list(frames.keys()))
        split_idx = int(len(frame_ids) * 0.8)
        
        for i, fid in enumerate(frame_ids):
            real_lidar_path = find_osdar_lidar_file(fid, ann_file, file_index)
            if not real_lidar_path: continue

            info = dict(sample_idx=fid, scene_id=scene_id, lidar_path=real_lidar_path, pose=None)
            if i < split_idx: infos_train.append(info)
            else: infos_val.append(info)

    print(f"-> OSDaR Result: {len(infos_train)} train, {len(infos_val)} val")
    if len(infos_train) > 0:
        with open(os.path.join(out_dir, 'osdar23_infos_train.pkl'), 'wb') as f: pickle.dump(infos_train, f)
        with open(os.path.join(out_dir, 'osdar23_infos_val.pkl'), 'wb') as f: pickle.dump(infos_val, f)

# ==========================================
# 3. SOSDaR 逻辑 (全新重写: 目录扫描模式)
# ==========================================
def get_sorted_pcd_files(json_path):
    """
    扫描 JSON 同级目录下的 streams/pandar64 (或 lidar)
    返回按文件名排序的绝对路径列表
    """
    scene_dir = json_path.parent
    
    # 尝试多种可能的雷达目录名
    candidates = [
        scene_dir / 'streams' / 'pandar64',
        scene_dir / 'streams' / 'lidar',
        scene_dir / 'streams' / 'tele15',
        scene_dir / 'lidar'
    ]
    
    lidar_dir = None
    for d in candidates:
        if d.exists() and d.is_dir():
            lidar_dir = d
            break
            
    if not lidar_dir:
        # print(f"Warning: No lidar dir found near {json_path.name}")
        return []
        
    # 获取所有 .pcd 文件并按文件名排序 (关键!)
    # 时间戳命名文件排序后就是时间顺序
    pcd_files = sorted(list(lidar_dir.glob("*.pcd")), key=lambda x: x.name)
    return [str(p) for p in pcd_files]

def create_sosdar_infos(root_path, out_dir):
    print(f"\n[SOSDaR] Processing with Directory Scanning...")
    
    infos_train = []
    # 查找所有JSON
    all_files = list(Path(root_path).rglob("*.json"))
    print(f"Found {len(all_files)} JSON files.")
    
    valid_scenes = 0
    
    for ann_file in all_files:
        try:
            with open(ann_file, 'r') as f: data = json.load(f)
        except: continue
            
        if 'openlabel' not in data: continue
        
        frames = data['openlabel'].get('frames', {})
        if not frames: continue
        
        # 1. 获取排序后的物理文件列表
        pcd_files = get_sorted_pcd_files(ann_file)
        
        if not pcd_files:
            continue
            
        # 2. 获取排序后的 Frame IDs (转为int排序)
        # SOSDaR的ID通常是 "0", "1", "2"...
        try:
            sorted_fids = sorted(frames.keys(), key=lambda x: int(x))
        except:
            sorted_fids = sorted(frames.keys())
            
        # 3. 执行映射 (Frame Index -> File Index)
        # 假设 Frame ID 0 对应 第1个文件，Frame ID 1 对应 第2个文件...
        # 即使 Frame ID 不连续，通常也意味着文件索引的对应
        
        matched_count = 0
        for i, fid in enumerate(sorted_fids):
            # 这是一个强假设：JSON里的第i帧对应文件夹里的第i个文件
            # 如果文件数量少于帧数，说明数据缺失，跳过越界的帧
            if i >= len(pcd_files):
                break
                
            real_path = pcd_files[i]
            
            # 也可以尝试用 Frame ID (int) 直接作为索引
            # idx = int(fid)
            # if idx < len(pcd_files): real_path = pcd_files[idx]
            
            info = dict()
            info['sample_idx'] = fid
            info['lidar_path'] = real_path
            info['pose'] = None # SOSDaR 只有几何信息，没有位姿
            
            infos_train.append(info)
            matched_count += 1
            
        if matched_count > 0:
            valid_scenes += 1
            # print(f"  Matched {ann_file.stem}: {matched_count} frames")
            
    print(f"-> SOSDaR Result: {len(infos_train)} samples from {valid_scenes} scenes")
    
    if len(infos_train) > 0:
        with open(os.path.join(out_dir, 'sosdar24_infos_train.pkl'), 'wb') as f:
            pickle.dump(infos_train, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osdar-root', required=True)
    parser.add_argument('--sosdar-root', required=True)
    args = parser.parse_args()
    
    create_osdar_infos(args.osdar_root, args.osdar_root)
    create_sosdar_infos(args.sosdar_root, args.sosdar_root)

if __name__ == '__main__':
    main()