# configs/_base_/model.py

# [核心修改 1] 改为 0.125，确保 grid_size=800 是 8 的倍数
voxel_size = [0.125, 0.125, 4]

model = dict(
    type='RailFusionNet',

    # 1. 体素化
    voxel_layer=dict(
        max_num_points=32, 
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),

    # 2. 体素编码
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[-50, -50, -5, 50, 50, 3]),

    # 3. 中间编码
    # [核心修改 2] 100 / 0.125 = 800
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[800, 800]), 

    # 4. Backbone (SECOND)
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),

    # 5. Neck (SECONDFPN)
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # 6. 障碍物检测头
    bbox_head=dict(
        type='RailCenterHead', 
        in_channels=384, 
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=1, class_names=['obstacle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            # [核心修改 3] 同步改为 0.125
            voxel_size=[0.125, 0.125]
        ),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    ),

    # 7. 轨道检测头
    rail_head=dict(
        type='PolyHead',
        in_channels=384, 
        num_polys=2,
        num_control_points=20,
        loss_poly=dict(type='ChamferDistanceLoss', loss_weight=1.0) 
    ),

    # 8. 训练配置
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            # [核心修改 4] 同步改为 800
            grid_size=[800, 800, 1],
            voxel_size=[0.125, 0.125, 4],
            out_size_factor=4, 
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
    ),

    # 9. 测试配置
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            # [核心修改 5] 同步改为 0.125
            voxel_size=[0.125, 0.125],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
        )
    )
)