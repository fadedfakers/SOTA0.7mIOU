import logging
import os
import sys
import torch.distributed as dist
from mmcv.utils import get_logger

def get_root_logger(log_file=None, log_level=logging.INFO, name='rail_bev'):
    """
    获取全局唯一的 Logger 实例 (基于 MMCV 封装)
    Args:
        log_file (str): 日志保存路径
        log_level (int): 日志级别
        name (str): Logger 名称
    """
    # 仅在主进程 (Rank 0) 打印日志，避免多卡训练时刷屏
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
        
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    if rank != 0:
        logger.setLevel('ERROR') # 从进程只记录错误

    # 格式化器设置 (如果在 get_logger 中未完全满足需求可在此补充)
    # MMCV 的 get_logger 已经配置得比较好了
    
    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """快捷打印函数"""
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, None, '
            f'or "silent", but got {type(logger)}')