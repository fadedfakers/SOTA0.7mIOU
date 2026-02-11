import pickle
import numpy as np

# 加载你的训练数据
pkl_path = '/root/autodl-tmp/FOD/data/osdar23_infos_train.pkl'
print(f"Loading {pkl_path}...")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

counts = {0: 0, 1: 0, 2: 0}
names = {0: "Car", 1: "Pedestrian", 2: "Obstacle"}

for info in data:
    labels = info['annos']['gt_labels_3d']
    for label in labels:
        if label in counts:
            counts[label] += 1

print("\n📊 Class Distribution in Training Set:")
print(f"  🚗 Car (ID 0):        {counts[0]}")
print(f"  🚶 Pedestrian (ID 1): {counts[1]}")
print(f"  📦 Obstacle (ID 2):   {counts[2]}")
print(f"  Total Objects:       {sum(counts.values())}")