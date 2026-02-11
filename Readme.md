这是一份为您定制的详细 `README.md` 文档。它总结了我们从 v1.0 到 v2.0 的所有重构工作，涵盖了架构设计、环境搭建、数据准备以及核心的**两阶段训练策略**。

建议您将此文件保存为项目根目录下的 `README.md`。

---

# Rail-BEV v2.0: Spatio-Temporal Fusion for Rail Obstacle Detection

## 📖 项目简介 (Introduction)

**Rail-BEV v2.0** 是针对轨道交通场景定制的 3D 感知系统，旨在解决列车前向障碍物检测与轨道线几何重建问题。

针对 v1.0 版本中存在的点云稀疏、训练策略失效及几何增强错位等问题，v2.0 进行了彻底的**架构重构**。本项目基于 `MMDetection3D` 框架开发，引入了 **时序多帧融合 (Temporal Fusion)** 和 **SOSDaR/OSDaR 双域联合训练** 策略，显著提升了在真实复杂场景下的检测 AP 和轨道分割 IoU。

### 🌟 v2.0 核心特性 (Key Features)

1. **时序多帧融合 (Temporal Fusion)**:
* 引入 `ConvGRU` 时序模块，利用 `Odom` 矩阵将过去 3 帧 () 的稀疏点云对齐至当前帧。
* 有效解决了 LiDAR 线束稀疏导致的漏检问题，不仅看“哪里有点”，还能看“点怎么动”。


2. **几何-视觉同步增强 (Sync-Geometry Augmentation)**:
* 修复了旧版 `transforms.py` 中“图片旋转但轨道标签不旋转”的致命 Bug。
* 实现了 `rotate_poly3d`，确保数据增强过程中 3D 轨道控制点与点云/图像严格同步。


3. **两阶段训练策略 (Two-Stage Training)**:
* **Phase 1 (Geometry):** 利用 **SOSDaR** 仿真数据完美的几何标注，预训练 Backbone 和 Rail Head。
* **Phase 2 (Temporal):** 加载预训练权重，在 **OSDaR23** 真实数据上进行时序检测微调。


4. **无 Anchor 动态检测头**:
* 采用 `CenterHead` 替代传统 Anchor-based 方法，更适应轨道异形障碍物。
* 轨道头重构为 `PolyHead`，直接回归 3D 控制点并使用 **Chamfer Distance Loss**。



---

## 🛠️ 环境安装 (Installation)

本项目依赖 `PyTorch` 和 `MMDetection3D`。建议在 `AutoDL` 提供的镜像基础上配置。

```bash
# 1. 创建虚拟环境
conda create -n railbev python=3.8 -y
conda activate railbev

# 2. 安装 PyTorch (根据您的 CUDA 版本调整，推荐 CUDA 11.3)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3. 安装 MMCV 和 MMDetection
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

# 4. 安装 MMDetection3D
pip install mmdet3d==1.0.0rc4

# 5. 安装其他依赖
pip install raillabel open3d tensorboard opencv-python

```

---

## 📂 数据准备 (Data Preparation)

请确保数据已挂载至 `AutoDL` 的 `tmp` 目录，目录结构应严格如下：

```text
/root/autodl-tmp/FOD/
├── data/                       # OSDaR23 (真实域)
│   ├── OSDaR23_LiDAR_Point_Clouds/
│   ├── OSDaR23_Image_Semantic/
│   └── annotation/             # .json 标注文件
└── SOSDaR24/                   # SOSDaR (仿真域)
    ├── frames/
    └── *.json                  # OpenLABEL 格式标注

```

### 生成数据索引

运行以下脚本，解析原始 JSON/OpenLABEL 数据，生成训练所需的 `.pkl` 索引文件：

```bash
python tools/create_data.py --osdar-root /root/autodl-tmp/FOD/data --sosdar-root /root/autodl-tmp/FOD/SOSDaR24

```

*成功运行后，将在对应目录下生成 `osdar23_infos_train.pkl` 和 `sosdar24_infos_train.pkl`。*

---

## 🚀 训练指南 (Training Guide)

v2.0 采用 **"先几何，后时序"** 的两阶段训练策略，以最大化利用仿真数据的几何精度和真实数据的时序特征。

### 第一阶段：SOSDaR 几何预训练 (Phase 1)

* **目标**: 让 Backbone 学会提取稳健的轨道几何特征，利用仿真数据量大、标注准的优势。
* **配置**: `configs/sosdar_geometry.py`
* **增强**: 开启高强度的几何旋转增强 (+/- 45度)。

```bash
# 单卡训练
python tools/train.py configs/sosdar_geometry.py --work-dir work_dirs/phase1_geometry

# 多卡训练 (例如 4 卡)
bash ./tools/dist_train.sh configs/sosdar_geometry.py 4 --work-dir work_dirs/phase1_geometry

```

### 第二阶段：OSDaR23 时序微调 (Phase 2)

* **目标**: 加载 Phase 1 权重，开启时序融合 (`frames_num=4`)，适应真实传感器噪声。
* **配置**: `configs/osdar23_temporal.py`
* **注意**: 需先修改配置中的 `load_from` 路径。

1. 修改 `configs/osdar23_temporal.py`:
```python
# 指向 Phase 1 训练好的最佳权重
load_from = 'work_dirs/phase1_geometry/latest.pth' 

```


2. 启动微调：
```bash
python tools/train.py configs/osdar23_temporal.py --work-dir work_dirs/phase2_temporal

```



### 监控训练

使用 TensorBoard 实时查看 Loss 曲线和 GPU 状态：

```bash
tensorboard --logdir work_dirs/

```

---

## 📊 评估与可视化 (Evaluation & Viz)

### 1. 计算指标 (Benchmark)

生成结果文件并计算 mAP 和 Chamfer Distance：

```bash
python tools/test.py configs/osdar23_temporal.py work_dirs/phase2_temporal/latest.pth --eval bbox

```

### 2. 3D 轨道重投影 (Visualization)

将预测的 3D 轨道投影回 2D 图像，直观验证几何对齐效果（红色为预测，绿色为真值）：

```bash
python tools/visualize.py configs/osdar23_temporal.py work_dirs/phase2_temporal/latest.pth --out-dir vis_results

```

---

## 🏗️ 项目结构 (Project Structure)

```text
Rail-BEV-v2.0/
├── configs/
│   ├── _base_/
│   │   ├── dataset.py        # 定义 RailDataset, Pipeline (含 OSDaR/SOSDaR)
│   │   ├── model.py          # 定义 RailFusionNet, PolyHead, CenterHead
│   │   └── schedule.py       # 优化器与 LR 策略
│   ├── sosdar_geometry.py    # Phase 1 配置文件
│   └── osdar23_temporal.py   # Phase 2 配置文件
├── data/
│   ├── osdar23_adapter.py    # [核心] 处理时序点云堆叠 (Odom对齐)
│   ├── sosdar_adapter.py     # 解析 OpenLABEL 格式
│   ├── transforms.py         # [修复] 含 rotate_poly3d 增强
│   └── sampler.py            # 类别平衡采样器
├── models/
│   ├── detectors/rail_fusion_net.py
│   ├── backbones/pillar_net.py # 支持 5D 输入 (x,y,z,i,dt)
│   ├── necks/temporal_fusion.py # ConvGRU 时序融合
│   └── heads/
│       ├── center_head.py    # 障碍物检测
│       └── poly_head.py      # 轨道控制点回归
├── tools/
│   ├── train.py              # [修复] 解除参数冻结，支持 DDP
│   ├── create_data.py        # 数据预处理
│   └── ...
└── utils/
    ├── geometry_ops.py       # 几何变换数学库
    └── metric_ops.py         # Chamfer Loss 与 IoU 计算

```

---

## 📝 常见问题 (FAQ)

**Q: 为什么 Phase 2 训练初期 Loss 会突然升高？**
A: 这是正常的。因为 Phase 1 是纯几何训练，切换到 Phase 2 后，模型需要适应真实域的激光雷达噪声和时序特征的引入。建议在 Phase 2 使用较小的学习率 (`lr=2e-4`)。

**Q: 报错 `RuntimeError: CUDA out of memory` 怎么办？**
A: BEV 模型显存占用较大。尝试在 `configs/_base_/dataset.py` 中减小 `samples_per_gpu` (例如从 4 改为 2)，或者减小 `frames_num` (从 4 改为 3)。

**Q: 可视化结果中轨道没有完全贴合铁轨？**
A: 检查 `data/transforms.py` 中的 `rotate_poly3d` 是否生效。另外，真实场景的地面高度变化可能导致投影误差，PolyHead 预测的是 3D 空间曲线，2D 投影仅供参考。

---

## 📧 联系与致谢

项目基于 Vicomtech OSDaR23 数据集与 MMDetection3D 开发。
如有代码问题，请提交 Issue 或检查 `logs/` 目录下的详细日志。