import json
import os
import glob
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# === 配置你的数据路径 ===
ROOT_PATH = '/root/autodl-tmp/FOD/data/'

# 我们的关键词表 (测试用)
KEYWORD_MAP = {
    'car': 'car', 'van': 'car', 'truck': 'car', 'bus': 'car', 'tram': 'car', 'train': 'car', 'railvehicle': 'car', 'vehicle': 'car',
    'pedestrian': 'pedestrian', 'person': 'pedestrian', 'cyclist': 'pedestrian', 'rider': 'pedestrian', 'human': 'pedestrian',
    'obstacle': 'obstacle', 'box': 'obstacle', 'animal': 'obstacle', 'stone': 'obstacle', 'signal': 'obstacle', 'pole': 'obstacle', 'catenary': 'obstacle', 'buffer': 'obstacle',
    'unknown': 'obstacle' # 怀疑之前是因为这个!
}

def check_data():
    all_types = Counter()
    unmatched_types = Counter()
    
    # 找所有 JSON
    json_files = list(Path(ROOT_PATH).rglob("*_labels.json"))
    print(f"Found {len(json_files)} JSON files. Scanning types...")
    
    for jf in tqdm(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            frames = data.get('openlabel', {}).get('frames', {})
            for fid, frame in frames.items():
                objects = frame.get('objects', {})
                for oid, obj in objects.items():
                    # 获取原始 Type
                    raw_type = obj.get('type', 'NO_TYPE_FIELD')
                    all_types[raw_type] += 1
                    
                    # 测试匹配
                    lower_type = str(raw_type).lower()
                    matched = False
                    for key in KEYWORD_MAP:
                        if key in lower_type:
                            matched = True
                            break
                    
                    if not matched:
                        unmatched_types[raw_type] += 1
                        
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    print("\n" + "="*40)
    print("📊 TOP 20 OBJECT TYPES FOUND:")
    print("="*40)
    for t, c in all_types.most_common(20):
        lower_t = str(t).lower()
        status = "✅ MATCH"
        match_k = ""
        for k in KEYWORD_MAP:
            if k in lower_t:
                match_k = k
                break
        if not match_k: status = "❌ UNMATCHED"
        else: status = f"✅ MATCH ({match_k})"
        
        print(f"{c:<6} | {str(t):<30} | {status}")

    print("\n" + "="*40)
    print("⚠️ TOP UNMATCHED TYPES (Potentially missing cars/peds):")
    print("="*40)
    for t, c in unmatched_types.most_common(20):
        print(f"{c:<6} | {str(t):<30}")

if __name__ == '__main__':
    check_data()