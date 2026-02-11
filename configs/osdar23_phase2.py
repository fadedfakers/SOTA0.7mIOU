_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/default_runtime.py'
]

# ==========================================================
# [战术阶段三：BEV 分割微调] - FINAL
# ==========================================================
# 策略：保持检测头不变，替换轨道头为 BEV 分割，解决乱飞问题
# ----------------------------------------------------------

point_cloud_range = [0, -44.8, -5, 204.8, 44.8, 10]
voxel_size = [0.4, 0.4, 0.2] 
grid_size = [512, 224, 75] # BEV 特征图大小对应 [W=224, H=512]

dataset_type = 'SOSDaRDataset' 
data_root = '/root/autodl-tmp/FOD/data/'
class_names = ['car', 'pedestrian', 'obstacle'] 

input_modality = dict(use_lidar=True, use_camera=True)
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadSOSDaRPCD', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    
    # [关键配置] 调用 sosdar_adapter.py 中的 FormatPoly
    # bev_size 对应 Mask 的分辨率 (H, W) = (224, 512) 还是 (512, 224)?
    # MMDetection3D 的 BEV 特征通常是 [H, W] = [512, 224] (基于 voxel_size 和 range 计算)
    # 这里的 bev_size 参数要传给 adapter 用于生成同样大小的 Mask
    dict(type='FormatPoly', bev_size=(224, 512), pc_range=point_cloud_range), 
    
    dict(type='Collect3D', 
         # [关键配置] 加入 'gt_masks' 字段，这是 BEV 分割的真值图片
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_masks'], 
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
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
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
    workers_per_gpu=4, 
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'osdar23_infos_train.pkl', 
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        filter_empty_gt=False 
    ),
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

model = dict(
    type='RailFusionNet', 
    
    # 基础权重 (Backbone)
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/sosdar_geometry/phase1_best.pth'),

    voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(90000, 120000) 
    ),

    voxel_encoder=dict(
        _delete_=True,
        type='HardSimpleVFE',
        num_features=4, 
    ),

    middle_encoder=dict(
        _delete_=True,
        type='SparseEncoder',
        in_channels=4, 
        sparse_shape=[75, 224, 512], 
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'
    ),

    backbone=dict(
        type='SECOND',
        in_channels=512, 
        out_channels=[64, 128, 256],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 2], 
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),

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
        frames_num=4,
        fusion_method='mvx' 
    ),
    
    bbox_head=dict(
        type='RailCenterHead', 
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['obstacle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-10, -50, -10, 210, 50, 10],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9  
        ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    ),
    
    # [核心修改] 替换为 BEV 分割头
    rail_head=dict(
        type='BEVSegHead',
        in_channels=384,    # 对应 Neck 的输出通道
        num_classes=1,      # 轨道只有一类
        loss_seg=dict(type='DiceLoss', loss_weight=2.0)
        # ⚠️ 注意：绝对不能有 num_polys, num_control_points, pc_range 等旧参数！
    ),

    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
        )
    ),
    
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-10, -50, -10, 210, 50, 10],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)

# 继续使用你已经训练好的权重 (epoch_12.pth)，这里面包含了强大的 Backbone 参数
load_from = 'work_dirs/osdar23_phase2/epoch_12.pth'

log_config = dict(
    interval=1,  
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])