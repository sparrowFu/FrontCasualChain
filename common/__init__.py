"""
共享工具和配置模块
"""
from .config import BaseConfig
from .metrics import AvgMeter, get_lr
from .dataset import (
    MSCOCOCaptionsDataset,
    load_mscoco_data,
    build_loaders,
    get_transforms
)
from .training import train_epoch, valid_epoch

# 为了向后兼容，创建别名
make_train_valid_dfs = load_mscoco_data

__all__ = [
    'BaseConfig',
    'AvgMeter',
    'get_lr',
    'MSCOCOCaptionsDataset',
    'load_mscoco_data',
    'make_train_valid_dfs',
    'build_loaders',
    'get_transforms',
    'train_epoch',
    'valid_epoch'
]
