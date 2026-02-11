import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys

# [修复] 确保项目根目录在 python path 中
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet3d import __version__
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger

import models
import data

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use', default=[0])
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    # [修复] 注入 seed 到 cfg 中
    cfg.seed = args.seed
    
    # [修复] 注入 gpu_ids 到 cfg 中
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.launcher == 'none' else range(8)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    
    logger.info(f'Building model from {args.config}...')
    
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    logger.info("Checking parameter gradients:")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_count += 1
        else:
            trainable_count += 1
    logger.info(f"Trainable params: {trainable_count}, Frozen params: {frozen_count}")
            
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
        
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet3d_version=__version__ + get_git_hash()[:7])

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta
    )

if __name__ == '__main__':
    main()