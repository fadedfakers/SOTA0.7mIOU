_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py'
]

# ==========================================================
# [1] 基础变量定义 (必须放在最前面！)
# ==========================================================
dataset_type = 'SOSDaRDataset'
data_root = '/root/autodl-tmp/FOD/SOSDaR24/'
class_names = ['car', 'pedestrian', 'obstacle'] 
input_modality = dict(use_lidar=True, use_camera=False)

# ==========================================================
# [2] 数据处理流水线
# ==========================================================
train_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    # dict(type='LoadAnnotations3D', ...), # 这个之前已经注掉了
    
    # =======================================================
    # [修改] ！！！必须注释掉下面这两个几何增强！！！
    # 否则点云转了，轨道没转，模型就疯了
    # =======================================================
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    
    # 范围过滤必须保留，这是为了切掉远处的杂点
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    
    dict(type='FormatPoly'), # 我们刚才加的
    
    dict(type='Collect3D', 
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'], 
         # flip 相关的 meta keys 其实也没用了，不过留着不报错
         meta_keys=['pts_filename', 'sample_idx', 'pcd_horizontal_flip', 
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
]

test_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', 
                 keys=['points'],
                 meta_keys=['pts_filename', 'sample_idx', 'pcd_horizontal_flip', 
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
        ])
]

# ==========================================================
# [3] 数据配置 (关键修改: workers_per_gpu=0)
# ==========================================================
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=0,  # <--- [核心修复] 改为 0 以解决 Autodl 卡死问题
    persistent_workers=False, # workers=0 时必须为 False
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sosdar24_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

# ==========================================================
# [4] 优化器配置
# ==========================================================
optimizer = dict(
    type='AdamW', 
    lr=0.003, 
    weight_decay=0.01
)

# 防止验证集坏数据导致训练中断，设为极大值跳过验证
evaluation = dict(interval=100)

# 注册自定义模块
custom_imports = dict(
    imports=['data.sosdar_adapter', 'models'], 
    allow_failed_imports=False
)

# ==========================================================
# [Phase 1 战略覆盖配置] 
# 使用 MMCV 字典合并机制，覆盖 _base_ 中的默认配置
# ==========================================================
model = dict(
    # 1. 强制单帧模式 (覆盖 _base_/model.py 中的时序配置)
    neck=dict(
        type='TemporalFusion', # 显式保留类型，防止被清空
        frames_num=1,          # 关键：单帧
        fusion_method='identity'
    ),
    # 2. 冻结检测头 Loss (切断梯度回传)
    bbox_head=dict(
        loss_cls=dict(loss_weight=0.0),
        loss_bbox=dict(loss_weight=0.0)
    ),
    # 3. 注入 Phase 1 特有的 RailHead 参数
    rail_head=dict(
        pc_range=[-50, -50, -5, 50, 50, 3] # 必须与 PointCloudRange 一致
    )
)