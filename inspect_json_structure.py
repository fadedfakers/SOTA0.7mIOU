import json
import os
import numpy as np

# 这是一个示例路径，脚本会自动找一个存在的 JSON 读取
SEARCH_ROOT = '/root/autodl-tmp/FOD/SOSDaR24/'

def find_first_json():
    # 找一个非空的 JSON 文件
    for root, dirs, files in os.walk(SEARCH_ROOT):
        for file in files:
            if file.endswith(".json"):
                return os.path.join(root, file)
    return None

def inspect():
    json_path = find_first_json()
    if not json_path:
        print("❌ 没找到任何 JSON 文件！")
        return

    print(f"📂 [侦探] 正在解剖文件: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('openlabel', {}).get('frames', {})
    if not frames:
        print("❌ JSON 居然是空的（没有 frames）！")
        return

    # 只看第一帧 (Key通常是 '0')
    first_frame_key = sorted(frames.keys(), key=lambda x: int(x))[0]
    frame_data = frames[first_frame_key]
    print(f"👀 [侦探] 正在检查第 {first_frame_key} 帧...")
    
    objects = frame_data.get('objects', {})
    print(f"📊 该帧包含 {len(objects)} 个对象")
    
    # 1. 打印所有出现过的 unique types
    all_types = set()
    for obj_id, obj in objects.items():
        all_types.add(obj.get('type', 'Unknown'))
    print(f"\n🧩 [关键线索] 发现的所有对象类型 (Types): {list(all_types)}")
    
    # 2. 深度搜索：谁肚子里有 'poly3d'？
    print("\n🔍 [深度搜索] 正在寻找包含 'poly3d' 的对象...")
    found_poly = False
    for obj_id, obj in objects.items():
        obj_data = obj.get('object_data', {})
        
        # 检查 poly3d
        has_poly = 'poly3d' in obj_data
        
        if has_poly:
            found_poly = True
            obj_type = obj.get('type', 'Unknown')
            print(f"\n✅ 找到目标！对象 ID: {obj_id}")
            print(f"   - 类型 (Type): '{obj_type}'")
            print(f"   - 数据结构 keys: {list(obj_data.keys())}")
            
            # 打印 poly3d 的具体值看看格式
            poly_val = obj_data['poly3d']
            print(f"   - poly3d 内容预览: {str(poly_val)[:200]} ...")
            
            # 如果这看起来像轨道，我们就破案了
            break
    
    if not found_poly:
        print("\n❌ 坏消息：在 'objects' 里没找到任何带 'poly3d' 的东西。")
        print("   可能轨道存储在 'contexts' 或 'relations' 字段里？")
        # 检查一下 frames 同级的其他字段
        print(f"   Frame 里的其他字段: {list(frame_data.keys())}")

if __name__ == '__main__':
    inspect()