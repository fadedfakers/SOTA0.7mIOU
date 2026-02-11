import os
import json
import pickle
import numpy as np
import glob
from tqdm import tqdm

# =================配置区域=================
DATA_ROOT = '/root/autodl-tmp/FOD/SOSDaR24/'
OUTPUT_PKL = os.path.join(DATA_ROOT, 'sosdar24_infos_train.pkl')
# ==========================================

def parse_openlabel_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    openlabel = data.get('openlabel', {})
    frames = openlabel.get('frames', {})
    
    if not frames: return []

    # 1. 获取该场景下所有的 PCD 文件并排序
    scene_dir = os.path.dirname(json_path)
    scene_name = os.path.basename(scene_dir)
    pcd_dir = os.path.join(scene_dir, 'streams/pandar64')
    
    if not os.path.exists(pcd_dir): return []

    # 按时间戳排序
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    sorted_frame_keys = sorted(frames.keys(), key=lambda x: int(x))
    
    parsed_infos = []
    
    for i, frame_idx_str in enumerate(sorted_frame_keys):
        if i >= len(pcd_files): break
            
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
            
            # [核心逻辑修改] 
            # 不再信任 obj_info.get('type')，直接深入检查 poly3d
            
            # --- 1. 强力搜索轨道 (Polyline) ---
            poly_list = obj_data.get('poly3d', [])
            
            # 兼容性处理：有时它是dict，有时是list
            if isinstance(poly_list, dict): 
                poly_list = [poly_list]
            
            if poly_list:
                for poly_item in poly_list:
                    # 检查名字：只要包含 'rail' 就算轨道
                    poly_name = poly_item.get('name', '').lower()
                    
                    if 'rail' in poly_name:
                        val = poly_item.get('val')
                        if val:
                            # OpenLABEL: [x1, y1, z1, x2, y2, z2...]
                            try:
                                pts = np.array(val, dtype=np.float32).reshape(-1, 3)
                                # 简单的过滤：太短的线不要
                                if len(pts) > 2:
                                    gt_poly_3d.append(pts)
                            except:
                                pass

            # --- 2. 尝试提取 BBox (即使 Type 是 Unknown) ---
            # 如果 type 是 Unknown，我们尝试通过 cuboid 的属性来猜测，或者暂时跳过
            # Phase 1 重点是轨道，BBox 空着也没事。
            # 这里保留之前的逻辑，但放宽一点：如果有名叫 'cuboid' 的数据就提取
            cuboid_list = obj_data.get('cuboid', [])
            if isinstance(cuboid_list, dict): cuboid_list = [cuboid_list]
            
            if cuboid_list:
                for cuboid in cuboid_list:
                    vals = cuboid.get('val')
                    if vals and len(vals) >= 9:
                        try:
                            # 假设标准格式
                            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                            yaw = float(vals[5])
                            l, w, h = float(vals[6]), float(vals[7]), float(vals[8])
                            gt_bboxes_3d.append([x, y, z, l, w, h, yaw])
                            gt_labels_3d.append(0) # 默认为 Car/Obstacle
                        except:
                            pass

        info = {
            'sample_idx': frame_idx_str,
            'lidar_path': lidar_rel_path,
            'annos': {
                'gt_bboxes_3d': np.array(gt_bboxes_3d, dtype=np.float32) if gt_bboxes_3d else np.zeros((0, 7), dtype=np.float32),
                'gt_labels_3d': np.array(gt_labels_3d, dtype=np.long) if gt_labels_3d else np.zeros(0, dtype=np.long),
                'gt_poly_3d': gt_poly_3d
            }
        }
        parsed_infos.append(info)

    return parsed_infos

def main():
    json_files = glob.glob(os.path.join(DATA_ROOT, '*/*.json'))
    print(f"Found {len(json_files)} scenes.")
    
    all_infos = []
    
    print("🚀 Starting conversion (Fix Logic)...")
    for json_file in tqdm(json_files):
        try:
            infos = parse_openlabel_json(json_file)
            all_infos.extend(infos)
        except Exception as e:
            print(f"⚠️ Error parsing {json_file}: {e}")

    print(f"Total frames collected: {len(all_infos)}")
    
    # 统计
    n_rails = sum([len(x['annos']['gt_poly_3d']) for x in all_infos])
    print(f"📊 统计: 总共提取到 {n_rails} 条轨道样本。")
    
    if n_rails > 0:
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(all_infos, f)
        print(f"✅ Generated {OUTPUT_PKL} successfully!")
        print("🎉 现在去修改 sosdar_adapter.py，去掉 dummy_rail 吧！")
    else:
        print("❌ 还是没提取到！请检查脚本逻辑。")

if __name__ == '__main__':
    main()