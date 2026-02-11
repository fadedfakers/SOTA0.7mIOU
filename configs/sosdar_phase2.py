_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py'
]

# ==========================================================
# [1] OSDaR23 数据集设置
# ==========================================================
dataset_type = 'SOSDaRDataset' # 假设 OSDaR23 也兼容这个 Dataset 类（通常格式是一样的）
data_root = '/root/autodl-tmp/FOD/OSDaR23/' # <--- 核心修改：指向 OSDaR23
class_names = ['car', 'pedestrian', 'obstacle'] 

input_modality = dict(use_lidar=True, use_camera=True)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='FormatPoly'), 
    dict(type='Collect3D', 
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'], 
         meta_keys=['pts_filename', 'img_prefix', 'img_info', 'lidar2img',
                    'sample_idx', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                    'box_mode_3d', 'box_type_3d'])
]

test_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
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
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', 
                 keys=['points', 'img'],
                 meta_keys=['pts_filename', 'img_prefix', 'img_info', 'lidar2img',
                            'sample_idx', 'pcd_horizontal_flip', 
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d'])
        ])
]

data = dict(
    samples_per_gpu=4, 
    workers_per_gpu=0,
    persistent_workers=False,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # [核心修改] 这里的 pkl 文件名需要确认！
        # 请去你的 OSDaR23 目录确认一下是不是叫 osdar23_infos_train.pkl
        ann_file=data_root + 'osdar23_infos_train.pkl', 
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
evaluation = dict(interval=100)
custom_imports = dict(imports=['data.sosdar_adapter', 'models'], allow_failed_imports=False)

# ==========================================================
# [2] 模型配置 (融合 + 检测 + 几何底座)
# ==========================================================
model = dict(
    type='RailFusionNet', 
    
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    neck=dict(
        type='TemporalFusion',
        frames_num=1,          
        fusion_method='mvx' 
    ),
    
    bbox_head=dict(
        loss_cls=dict(loss_weight=1.0),
        loss_bbox=dict(loss_weight=1.0)
    ),
    
    rail_head=dict(
        pc_range=[-50, -50, -5, 50, 50, 3] 
    )
)

# [核心修改] 加载 SOSDaR 上练好的几何权重！
# 这是 "2+1" 战术的精髓：把 SOSDaR 学到的几何知识迁移到 OSDaR 上
#load_from = 'work_dirs/sosdar_geometry/phase1_best.pth'
load_from = None