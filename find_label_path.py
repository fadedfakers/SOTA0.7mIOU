import json
import os
from pathlib import Path

# === 配置路径 ===
ROOT_PATH = '/root/autodl-tmp/FOD/data/'

def find_label_location():
    # 随便找一个 JSON 文件
    json_files = list(Path(ROOT_PATH).rglob("*_labels.json"))
    if not json_files:
        print("❌ No JSON files found!")
        return
    
    target_file = json_files[0]
    print(f"🕵️ Inspecting file: {target_file.name}")
    
    with open(target_file, 'r') as f:
        data = json.load(f)
        
    frames = data.get('openlabel', {}).get('frames', {})
    
    # 搜索前几帧
    print("\n🔍 Searching for keywords (car, vehicle, pedestrian, obstacle)...")
    found_paths = []
    
    # 递归搜索函数
    def recursive_search(d, path_str, keywords):
        if isinstance(d, dict):
            for k, v in d.items():
                recursive_search(v, f"{path_str} -> {k}", keywords)
        elif isinstance(d, list):
            for i, v in enumerate(d):
                recursive_search(v, f"{path_str}[{i}]", keywords)
        elif isinstance(d, str):
            # 检查是否包含关键词
            val_lower = d.lower()
            for kw in keywords:
                if kw in val_lower:
                    print(f"  ✅ FOUND '{kw}' in value: '{d}'")
                    print(f"     📍 Path: {path_str}")
                    return

    keywords = ['car', 'vehicle', 'pedestrian', 'person', 'obstacle', 'signal']
    
    # 只检查第一个非空帧的对象
    for fid, frame in frames.items():
        objects = frame.get('objects', {})
        if not objects: continue
        
        print(f"Checking Frame {fid} ({len(objects)} objects)...")
        
        # 只检查前 3 个对象，避免刷屏
        count = 0
        for oid, obj in objects.items():
            print(f"\n--- Object ID: {oid} ---")
            # 打印对象的第一层键，帮我们从宏观看看
            print(f"    Top-level keys: {list(obj.keys())}")
            
            # 开始深挖
            recursive_search(obj, "obj", keywords)
            
            count += 1
            if count >= 3: break
        
        # 查完一帧就跑，这就够了
        break

if __name__ == '__main__':
    find_label_location()