import torch
# 加载 Phase 1 权重
ckpt = torch.load('work_dirs/sosdar_geometry/phase1_best.pth')
print("Phase 1 Keys:", [k for k in ckpt['state_dict'].keys() if 'rail_head' in k][:5])

# 加载 Phase 2 刚跑完的权重
ckpt2 = torch.load('work_dirs/osdar23_phase2/epoch_12.pth')
print("Phase 2 Keys:", [k for k in ckpt2['state_dict'].keys() if 'rail_head' in k][:5])