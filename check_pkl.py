# 创建一个名为 check_pkl.py 的文件
import mmcv
import numpy as np

# 读取你的标注文件
data = mmcv.load('/root/autodl-tmp/FOD/SOSDaR24/sosdar24_infos_train.pkl')
info = data[0] # 看第一帧数据

print("="*50)
print("Keys inside info:", info.keys())
print("-" * 20)
# 寻找可能存放轨道的字段，通常是 'annos', 'rails', 'lanes' 之类的
if 'annos' in info:
    print("Keys inside annos:", info['annos'].keys())
    # 检查是否有 gt_poly_3d 或 similar
    for key in info['annos']:
        print(f"  {key}: {type(info['annos'][key])}")
print("="*50)