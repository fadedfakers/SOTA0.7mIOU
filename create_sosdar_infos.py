import os
import json
import pickle
import numpy as np
import glob
from tqdm import tqdm

# =================配置区域=================
DATA_ROOT = '/root/autodl-tmp/FOD/SOSDaR24/'
OUTPUT_PKL = os.path.join(DATA_ROOT, 'sosdar24_infos_train.pkl')

# 类别映射表 (根据 OpenLABEL 常见定义)
CLASS_MAP = {
    'Car': 'car', 'Van': 'car', 'Truck': 'car',
    'Pedestrian': 'pedestrian', 'Cyclist': 'pedestrian', 'Person': 'pedestrian',
    'Obstacle': 'obstacle', 'Box': 'obstacle', 'Rock': 'obstacle'
}

# 轨道相关的关键词 (脚本会自动搜索包含这些词的类别)
RAIL_KEYWORDS = ['rail', 'track', 'poly']
# ==========================================

def parse_openlabel_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    openlabel = data.get('openlabel', {})
    frames = openlabel.get('frames', {})
    
    if not frames:
        return []

    # 1. 获取该场景下所有的 PCD 文件并排序
    scene_dir = os.path.dirname(json_path)
    scene_name = os.path.basename(scene_dir)
    pcd_dir = os.path.join(scene_dir, 'streams/pandar64')
    
    if not os.path.exists(pcd_dir):
        return []

    # 获取所有 .pcd 文件名并排序 (假设文件名是时间戳，排序即为时间顺序)
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    
    # 检查帧数匹配情况
    # 注意：有时候 JSON 帧数可能比 PCD 文件少或多，我们取交集或最小长度
    # 但通常 Keys 是 "0", "1", ... 
    
    parsed_infos = []
    
    # 遍历 JSON 中的每一帧
    # 我们假设 Key "0" 对应 pcd_files[0], Key "1" 对应 pcd_files[1]...
    sorted_frame_keys = sorted(frames.keys(), key=lambda x: int(x))
    
    for i, frame_idx_str in enumerate(sorted_frame_keys):
        if i >= len(pcd_files):
            break # PCD 文件不够了
            
        pcd_filename = pcd_files[i]
        frame_data = frames[frame_idx_str]
        
        # 构造相对路径
        lidar_rel_path = f"{scene_name}/streams/pandar64/{pcd_filename}"
        
        # --- 提取标注 ---
        objects = frame_data.get('objects', {})
        
        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_poly_3d = [] # 轨道线
        
        for obj_id, obj_info in objects.items():
            obj_data = obj_info.get('object_data', {})
            obj_type = obj_info.get('type', 'Unknown')
            
            # (A) 处理障碍物/车辆 (Cuboid)
            # 需要匹配 CLASS_MAP 中的 key
            matched_type = None
            for key in CLASS_MAP:
                if key.lower() in obj_type.lower():
                    matched_type = CLASS_MAP[key]
                    break
            
            if matched_type:
                cuboid = obj_data.get('cuboid', {})
                if not cuboid: 
                    cuboid_list = obj_data.get('cuboid', [])
                    if isinstance(cuboid_list, list) and len(cuboid_list) > 0: cuboid = cuboid_list[0]
                
                vals = cuboid.get('val')
                if vals:
                    try:
                        # OpenLABEL 标准: x, y, z, r, p, yaw, w, l, h (或者 l, w, h)
                        # 我们假设是 x, y, z, ..., yaw, l, w, h
                        # 根据 SOSDaR 样本，通常 vals 长度为 9 或 10
                        # 0:cx, 1:cy, 2:cz, 3:rx, 4:ry, 5:rz(yaw), 6:sx, 7:sy, 8:sz
                        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                        yaw = float(vals[5])
                        
                        # 尺寸映射需要小心，通常是 6,7,8 或者 7,8,9
                        # 这里先假设 7=l, 6=w, 8=h (根据常见定义，cx,cy,cz, rx,ry,rz, sx,sy,sz)
                        # MMDetection: x, y, z, dx, dy, dz, yaw
                        l = float(vals[6]) # sx (length)
                        w = float(vals[7]) # sy (width)
                        h = float(vals[8]) # sz (height)
                        
                        gt_bboxes_3d.append([x, y, z, l, w, h, yaw])
                        
                        # 简单映射 label: car=0, pedestrian=1, obstacle=2
                        label_id = 0
                        if matched_type == 'pedestrian': label_id = 1
                        elif matched_type == 'obstacle': label_id = 2
                        gt_labels_3d.append(label_id)
                    except:
                        pass

            # (B) 处理轨道 (Polyline)
            is_rail = any(k in obj_type.lower() for k in RAIL_KEYWORDS)
            if is_rail:
                poly = obj_data.get('poly3d')
                if not poly:
                    poly_list = obj_data.get('poly3d', [])
                    if isinstance(poly_list, list) and len(poly_list) > 0: poly = poly_list[0]
                
                if poly:
                    val = poly.get('val')
                    if val:
                        # OpenLABEL poly3d val 是一维数组 [x1, y1, z1, x2, y2, z2...]
                        pts = np.array(val, dtype=np.float32).reshape(-1, 3)
                        gt_poly_3d.append(pts)

        # 构造 Info
        info = {
            'sample_idx': frame_idx_str,
            'lidar_path': lidar_rel_path,
            'annos': {
                'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32) if gt_bboxes_3d else np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.array(gt_labels_3d, dtype=np.long) if gt_labels_3d else np.zeros(0, dtype=np.long),
                'gt_poly_3d': gt_poly_3d # list of numpy arrays
            }
        }
        parsed_infos.append(info)

    return parsed_infos

def main():
    # 递归查找所有 json 文件
    json_files = glob.glob(os.path.join(DATA_ROOT, '*/*.json'))
    print(f"Found {len(json_files)} scenes.")
    
    all_infos = []
    
    print("🚀 Starting conversion...")
    for json_file in tqdm(json_files):
        try:
            infos = parse_openlabel_json(json_file)
            all_infos.extend(infos)
        except Exception as e:
            print(f"⚠️ Error parsing {json_file}: {e}")

    print(f"Total frames collected: {len(all_infos)}")
    
    if len(all_infos) > 0:
        # 简单统计一下
        n_rails = sum([len(x['annos']['gt_poly_3d']) for x in all_infos])
        print(f"📊 统计: 总共提取到 {n_rails} 条轨道样本。")
        
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(all_infos, f)
        print(f"✅ Generated {OUTPUT_PKL} successfully!")
    else:
        print("❌ No info generated. Check data root.")

if __name__ == '__main__':
    main()