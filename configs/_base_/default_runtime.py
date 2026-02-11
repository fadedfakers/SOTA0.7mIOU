# [修复] 删除了 checkpoint_config，防止与 schedule.py 冲突
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# dist_params = dict(backend='nccl') # 如果单卡运行报错可以注释掉这行
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]