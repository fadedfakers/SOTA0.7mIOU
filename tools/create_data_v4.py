import pickle
import os
import json
import argparse
from pathlib import Path
import sys

def get_files_in_dir(dir_path):
    """获取目录下所有文件，返回排序后的列表"""
    if not os.path.exists(dir_path):
        return []
    return sorted([f for f in os.listdir(dir_path) if not f.startswith('.')])

def find_file_by_prefix(file_list, prefix):
    """在文件列表中查找以 prefix 开头的文件 (复刻 dataset.py 的 next(...) 逻辑)"""
    for f in file_list:
        if f.startswith(prefix):
            return f
    return None

def create_osdar_infos(root_path, out_dir):
    print(f"\n🚀 [OSDaR23] 启动 V4 生成脚本 (基于旧项目逻辑复刻)...")
    
    # 1. 扫描场景目录 (复刻 dataset.py)
    # 直接遍历 root 下的子文件夹，而不是递归找文件
    if not os.path.exists(root_path):
        print(f"❌ 路径不存在: {root_path}")
        return

    all_scenes = sorted([
        d for d in os.listdir(root_path) 
        if os.path.isdir(os.path.join(root_path, d)) and not d.startswith('.')
    ])
    
    print(f"📂 扫描到 {len(all_scenes)} 个场景文件夹")
    
    infos_train = []
    infos_val = []
    total_frames = 0
    valid_scenes_count = 0
    
    # 2. 遍历场景
    for scene_id in all_scenes:
        scene_dir = os.path.join(root_path, scene_id)
        
        # 寻找 JSON (通常是 scene_id_labels.json，但也可能是其他名字)
        # 优先匹配 dataset.py 中的命名规则
        json_path = os.path.join(scene_dir, f"{scene_id}_labels.json")
        if not os.path.exists(json_path):
            # 备选：找目录下唯一的 .json
            candidates = list(Path(scene_dir).glob("*.json"))
            if candidates:
                json_path = str(candidates[0])
            else:
                print(f"⚠️  [跳过] 场景 {scene_id}: 未找到 JSON")
                continue

        # 寻找 LiDAR 和 RGB 目录
        lidar_dir = os.path.join(scene_dir, "lidar")
        if not os.path.exists(lidar_dir):
            lidar_dir = os.path.join(scene_dir, "points") # 备选
        
        # 图像目录优先序
        img_dir_candidates = ["rgb_center", "rgb_highres_center", "image_02"]
        img_dir = None
        for cand in img_dir_candidates:
            d = os.path.join(scene_dir, cand)
            if os.path.exists(d):
                img_dir = d
                break
        
        if not os.path.exists(lidar_dir):
            # print(f"⚠️  [跳过] 场景 {scene_id}: 缺少 lidar 目录")
            continue

        # 3. 读取目录文件列表 (预加载以加速)
        pcd_all = get_files_in_dir(lidar_dir)
        img_all = get_files_in_dir(img_dir) if img_dir else []
        
        # 4. 解析 JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            print(f"❌ 读取 JSON 失败: {scene_id}")
            continue
            
        # 获取帧列表
        frames = data.get('openlabel', {}).get('frames', {})
        if not frames: frames = data.get('frames', {}) # 兼容旧格式
        
        if not frames: continue

        # 5. 标定文件路径 (Phase 2 需要)
        calib_path = os.path.join(scene_dir, "calibration.txt")
        if not os.path.exists(calib_path):
            calib_path = None # 允许为空，但建议有

        # 6. 核心匹配循环
        scene_valid_count = 0
        
        # 对 Frame ID 排序 (数字序)
        try:
            sorted_fids = sorted(frames.keys(), key=lambda x: int(x))
        except:
            sorted_fids = sorted(frames.keys())

        # 划分数据集 (每个场景前80%训练)
        split_idx = int(len(sorted_fids) * 0.8)

        for i, fid in enumerate(sorted_fids):
            # === [关键修复] 复刻 dataset.py 的匹配逻辑 ===
            try:
                fid_int = int(fid)
                # 策略 1: 补零匹配 (000_...) -> 这是您旧代码成功的关键
                prefix_pad = f"{fid_int:03d}_"
                
                # 策略 2: 原样匹配 (0_...) -> 兼容原生 OSDaR
                prefix_raw = f"{fid}_"
                
                pcd_f = find_file_by_prefix(pcd_all, prefix_pad)
                if not pcd_f:
                    pcd_f = find_file_by_prefix(pcd_all, prefix_raw)
                
                if not pcd_f: continue # 没点云就跳过

                # 找图片 (逻辑同上)
                img_f = None
                if img_dir:
                    img_f = find_file_by_prefix(img_all, prefix_pad)
                    if not img_f:
                        img_f = find_file_by_prefix(img_all, prefix_raw)
            
            except ValueError:
                continue

            # 构建 Info
            info = {
                'sample_idx': fid,
                'scene_id': scene_id,
                'lidar_path': os.path.join(lidar_dir, pcd_f),
                'img_path': os.path.join(img_dir, img_f) if img_f else None,
                'calib_path': calib_path,
                'pose': None
            }

            if i < split_idx:
                infos_train.append(info)
            else:
                infos_val.append(info)
            
            scene_valid_count += 1
        
        if scene_valid_count > 0:
            valid_scenes_count += 1
            total_frames += scene_valid_count
            # print(f"  - 场景 {scene_id}: 匹配 {scene_valid_count} 帧")

    print(f"\n✅ 处理完成!")
    print(f"   -> 有效场景: {valid_scenes_count}/{len(all_scenes)}")
    print(f"   -> 总帧数: {total_frames} (预期 ~899)")
    print(f"   -> 训练集: {len(infos_train)}")
    print(f"   -> 验证集: {len(infos_val)}")
    
    # 保存 .pkl
    with open(os.path.join(out_dir, 'osdar23_infos_train.pkl'), 'wb') as f:
        pickle.dump(infos_train, f)
    with open(os.path.join(out_dir, 'osdar23_infos_val.pkl'), 'wb') as f:
        pickle.dump(infos_val, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--osdar-root', required=True)
    parser.add_argument('--sosdar-root', required=False)
    args = parser.parse_args()
    
    create_osdar_infos(args.osdar_root, args.osdar_root)

if __name__ == '__main__':
    main()