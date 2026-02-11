import torch
from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads import CenterHead

@HEADS.register_module()
class RailCenterHead(CenterHead):
    """
    继承自 MMDetection3D 的 CenterHead。
    复用官方的 heatmap 生成、正负样本分配和 loss 计算逻辑。
    """
    def __init__(self, 
                 in_channels, 
                 tasks, 
                 grid_size=None, 
                 bbox_coder=None, 
                 common_heads=dict(), 
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'), 
                 loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25), 
                 separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64, 
                 num_hm_conv=2,  # 接收它，防止 Config 报错
                 norm_bbox=True, 
                 init_cfg=None,
                 **kwargs):
        
        # [核心修复] 父类 CenterHead 不接受 num_hm_conv，所以这里不传它
        # 如果需要调整卷积层数，请修改 separate_head 的配置
        super(RailCenterHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            # num_hm_conv=num_hm_conv, <--- 删掉了这一行
            norm_bbox=norm_bbox,
            init_cfg=init_cfg,
            **kwargs
        )