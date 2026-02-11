# 数据集类型
dataset_type = 'SOSDaRDataset'
# 数据根目录
data_root = '/root/autodl-tmp/FOD/SOSDaR24/'

# 训练数据处理管道
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    # [修复] 移除了 'with_poly_3d=True'，因为它会导致官方类报错
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'pedestrian', 'obstacle']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# 测试数据处理管道
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
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
                class_names=['car', 'pedestrian', 'obstacle'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# 数据加载配置
data = dict(
    samples_per_gpu=2,  # 显存如果不够可以改为 1
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # [关键] 指向您刚刚生成的真实 pkl 路径
        ann_file='/root/autodl-tmp/FOD/SOSDaR24/sosdar24_infos_train.pkl',
        pipeline=train_pipeline,
        classes=['car', 'pedestrian', 'obstacle'],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=False,
        filter_empty_gt=False  # SOSDaR Phase 1 没真值，不要过滤空数据
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/root/autodl-tmp/FOD/SOSDaR24/sosdar24_infos_train.pkl', # 暂时用 train 做 val
        pipeline=test_pipeline,
        classes=['car', 'pedestrian', 'obstacle'],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/root/autodl-tmp/FOD/SOSDaR24/sosdar24_infos_train.pkl',
        pipeline=test_pipeline,
        classes=['car', 'pedestrian', 'obstacle'],
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        test_mode=True
    )
)