import pickle
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 关键词映射表 (BBox)
# ==========================================
KEYWORD_MAP = {
    'car': 'car', 'van': 'car', 'truck': 'car', 'bus': 'car', 
    'tram': 'car', 'train': 'car', 'railvehicle': 'car', 'vehicle': 'car',
    'wagon': 'car', 'locomotive': 'car',
    'pedestrian': 'pedestrian', 'person': 'pedestrian', 
    'cyclist': 'pedestrian', 'rider': 'pedestrian', 'human': 'pedestrian',
    'obstacle': 'obstacle', 'box': 'obstacle', 'stone': 'obstacle', 
    'signal': 'obstacle', 'pole': 'obstacle', 'catenary': 'obstacle', 
    'buffer': 'obstacle', 'schaltkasten': 'obstacle'
}

TRAIN_CLASSES = ['car', 'pedestrian', 'obstacle']
CLASS_TO_ID = {name: i for i, name in enumerate(TRAIN_CLASSES)}

# ==========================================
# 2. 轨道线解析逻辑 (新增!)
# ==========================================
def parse_rails(frame_data):
    """
    解析轨道线 (Poly3d)
    返回: List[np.ndarray(N, 3)]
    """
    objects = frame_data.get('objects', {})
    gt_poly_3d = []
    
    for obj_id, obj_info in objects.items():
        obj_data = obj_info.get('object_data', {})
        
        # 1. 检查名字是否像轨道
        # OSDaR 中轨道通常叫 "Track", "Rail", "Gleis"
        name_str = ""
        
        # 尝试从各种地方找名字
        if 'name' in obj_info: name_str = obj_info['name']
        elif 'type' in obj_info: name_str = obj_info['type']
        
        # 这是一个宽松的过滤，稍后我们主要看它有没有 poly3d 数据
        is_rail_candidate = any(k in name_str.lower() for k in ['track', 'rail', 'gleis', 'lane'])
        
        # 2. 提取 Poly3d
        # OpenLABEL 结构: object_data -> poly3d (List)
        raw_polys = obj_data.get('poly3d', [])
        if isinstance(raw_polys, dict): raw_polys = [raw_polys]
        
        if not raw_polys:
            continue

        for poly in raw_polys:
            # 有些轨道标注藏在 poly3d 的 name 里
            poly_name = poly.get('name', '').lower()
            
            # 如果对象本身没名字，且 poly 也没名字，跳过 (防止提取到奇怪的线)
            if not is_rail_candidate and not any(k in poly_name for k in ['track', 'rail', 'gleis']):
                continue

            vals = poly.get('val')
            if not vals: continue
            
            try:
                # OpenLABEL poly3d val: [x1, y1, z1, x2, y2, z2, ...]
                points = np.array(vals, dtype=np.float32).reshape(-1, 3)
                
                # 简单的过滤：点太少不算线
                if points.shape[0] < 2: continue
                
                # 坐标系修正: OSDaR 这里的 poly3d 也是基于 Sensor 坐标系的?
                # 通常 LiDAR 点云不需要修正，直接用
                gt_poly_3d.append(points)
                
            except Exception:
                continue
                
    return gt_poly_3d

# ==========================================
# 3. 这里的 BBox 解析逻辑保持不变
# ==========================================
def parse_objects(frame_data):
    objects = frame_data.get('objects', {})
    gt_bboxes = []
    gt_labels = []
    gt_names = []
    
    for obj_id, obj_info in objects.items():
        obj_data = obj_info.get('object_data', {})
        raw_cuboids = obj_data.get('cuboid', [])
        if isinstance(raw_cuboids, dict): raw_cuboids = [raw_cuboids]
        
        if not raw_cuboids: continue

        matched_type = None
        target_cuboid = None
        
        for cuboid in raw_cuboids:
            name_str = cuboid.get('name', '').lower()
            for key, target_class in KEYWORD_MAP.items():
                if key in name_str:
                    matched_type = target_class
                    target_cuboid = cuboid
                    break
            if matched_type: break
        
        if not matched_type or not target_cuboid: continue
            
        vals = target_cuboid.get('val')
        if not vals: continue
            
        try:
            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
            yaw = float(vals[5])
            l, w, h = float(vals[6]), float(vals[7]), float(vals[8])
            
            if l <= 0.05 or w <= 0.05 or h <= 0.05: continue
            
            gt_bboxes.append([x, y, z, l, w, h, yaw])
            gt_labels.append(CLASS_TO_ID[matched_type])
            gt_names.append(matched_type)
        except Exception: continue

    if not gt_bboxes:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64), np.array([])
    
    return (np.array(gt_bboxes, dtype=np.float32), 
            np.array(gt_labels, dtype=np.int64),
            np.array(gt_names))

# ==========================================
# 4. 主生成函数 (整合 Box + Rail)
# ==========================================
def get_files_in_dir(dir_path):
    if not os.path.exists(dir_path): return []
    return sorted([f for f in os.listdir(dir_path) if not f.startswith('.')])

def find_file_by_prefix(file_list, prefix):
    for f in file_list:
        if f.startswith(prefix): return f
    return None

def create_osdar_infos_with_gt(root_path, out_dir):
    print(f"\n🚀 [OSDaR23] Generating Full GT Infos (BBox + Rails)...")
    
    if not os.path.exists(root_path): return

    all_scenes = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and not d.startswith('.')])
    
    infos_train = []
    infos_val = []
    total_objs = 0
    total_rails = 0 # 统计轨道数量
    
    for scene_id in tqdm(all_scenes, desc="Processing Scenes"):
        scene_dir = os.path.join(root_path, scene_id)
        json_path = os.path.join(scene_dir, f"{scene_id}_labels.json")
        if not os.path.exists(json_path):
            candidates = list(Path(scene_dir).glob("*.json"))
            if candidates: json_path = str(candidates[0])
            else: continue

        lidar_dir = os.path.join(scene_dir, "lidar")
        if not os.path.exists(lidar_dir): lidar_dir = os.path.join(scene_dir, "points")
        if not os.path.exists(lidar_dir): continue

        img_dir_candidates = ["rgb_center", "rgb_highres_center", "image_02"]
        img_dir = None
        for cand in img_dir_candidates:
            d = os.path.join(scene_dir, cand)
            if os.path.exists(d): img_dir = d; break
        
        pcd_all = get_files_in_dir(lidar_dir)
        img_all = get_files_in_dir(img_dir) if img_dir else []
        
        try:
            with open(json_path, 'r') as f: data = json.load(f)
        except: continue
            
        frames = data.get('openlabel', {}).get('frames', {})
        if not frames: frames = data.get('frames', {})
        if not frames: continue

        calib_path = os.path.join(scene_dir, "calibration.txt")
        if not os.path.exists(calib_path): calib_path = None

        try: sorted_fids = sorted(frames.keys(), key=lambda x: int(x))
        except: sorted_fids = sorted(frames.keys())
            
        split_idx = int(len(sorted_fids) * 0.8)

        for i, fid in enumerate(sorted_fids):
            try:
                fid_int = int(fid)
                prefix_pad = f"{fid_int:03d}_"
                prefix_raw = f"{fid}_"
                pcd_f = find_file_by_prefix(pcd_all, prefix_pad)
                if not pcd_f: pcd_f = find_file_by_prefix(pcd_all, prefix_raw)
                if not pcd_f: continue
                img_f = None
                if img_dir:
                    img_f = find_file_by_prefix(img_all, prefix_pad)
                    if not img_f: img_f = find_file_by_prefix(img_all, prefix_raw)
            except: continue
            
            # 1. 解析 BBox
            gt_bboxes_3d, gt_labels_3d, gt_names = parse_objects(frames[fid])
            
            # 2. 解析 Rails (新增!)
            gt_poly_3d = parse_rails(frames[fid])
            
            total_objs += len(gt_bboxes_3d)
            if len(gt_poly_3d) > 0: total_rails += 1

            info = {
                'sample_idx': fid,
                'scene_id': scene_id,
                'lidar_path': os.path.join(lidar_dir, pcd_f),
                'img_path': os.path.join(img_dir, img_f) if img_f else None,
                'calib_path': calib_path,
                'pose': None,
                'annos': {
                    'gt_bboxes_3d': gt_bboxes_3d,
                    'gt_labels_3d': gt_labels_3d,
                    'gt_names': gt_names,
                    'gt_poly_3d': gt_poly_3d # 保存轨道真值
                }
            }

            if i < split_idx: infos_train.append(info)
            else: infos_val.append(info)

    print(f"\n✅ Done!")
    print(f"   Total Objects (BBox): {total_objs}")
    print(f"   Frames with Rails:    {total_rails}")
    
    with open(os.path.join(out_dir, 'osdar23_infos_train.pkl'), 'wb') as f:
        pickle.dump(infos_train, f)
    with open(os.path.join(out_dir, 'osdar23_infos_val.pkl'), 'wb') as f:
        pickle.dump(infos_val, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--osdar-root', default='/root/autodl-tmp/FOD/data/', help='Path to OSDaR dataset root')
    parser.add_argument('--out-dir', default='/root/autodl-tmp/FOD/data/', help='Output directory for pkl files')
    args = parser.parse_args()
    create_osdar_infos_with_gt(args.osdar_root, args.out_dir)