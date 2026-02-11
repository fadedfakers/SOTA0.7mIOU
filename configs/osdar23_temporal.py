# configs/osdar23_temporal.py

_base_ = [
    './_base_/dataset.py',
    './_base_/model.py',
    './_base_/schedule.py'
]

# [实验元数据]
work_dir = './work_dirs/osdar23_temporal_v2'
experiment_name = 'v2.0_phase2_temporal_finetune'

# [SOTA 策略] 加载 SOSDaR 几何预训练权重
# 如果还没有预训练权重，注释掉此行
# load_from = './work_dirs/sosdar_geometry_pretrain/latest.pth' 

# [模型微调配置]
model = dict(
    type='RailFusionNet',
    
    # [关键] 适配时序输入的通道数 (x, y, z, i, dt) = 5
    backbone=dict(
        type='PillarFeatureNet',
        in_channels=5, 
        feat_channels=[64],
        # [微调策略] 冻结底层
        frozen_stages=1, 
    ),
    
    # [时序模块]
    neck=dict(
        type='TemporalFusion',
        frames_num=4,
        fusion_method='conv_gru' # GRU 对时序建模效果最好
    ),
    
    # [检测头] 真实域误报较多，调整 Loss 权重
    bbox_head=dict(
        type='CenterHead', # 假设使用 CenterHead
        loss_cls=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)
    ),
    
    # [几何头] 开启 Chamfer Loss
    # 这里定义的参数会被传入 PolyHead 的 __init__
    poly_head=dict(
        type='PolyHead',
        in_channels=64,   # 对应 Neck 的输出通道
        feat_channels=64, # 内部特征通道
        num_points=20,    # 每条轨道预测 20 个点
        
        # [关键] 损失函数配置，PolyHead 内部会调用 build_loss
        loss_poly=dict(
            type='ChamferDistanceLoss', 
            loss_weight=1.0
        )
    )
)

# [数据覆盖] 
# 覆盖 _base_，专注于 OSDaR23 真实域
data_root = '/root/autodl-tmp/FOD/data'
dataset_type = 'RailDataset'
class_names = ['pedestrian', 'car', 'obstacle', 'signal', 'buffer_stop']

train_pipeline = [
    # 这里的 pipeline 实际上是在 adapter 内部执行或调用的
    # 如果使用 mmdet3d 框架，这里定义的是 transforms
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925], # +/- 22.5度，比预训练保守
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0],
         update_poly3d=True), 

    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_poly_3d'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/osdar23_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        frames_num=4), # 确保时序开启
    val=dict(
        frames_num=4
    ),
    test=dict(
        frames_num=4
    )
)

# [训练策略优化]
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01) # 微调使用较小的 LR
runner = dict(max_epochs=12) # 微调轮数可以减少