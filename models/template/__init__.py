"""
新模型模块 - 添加新模型时参考此模板
"""
from .config import NewModelConfig
from .model import NewModel
from .train import train as train_new_model
from .evaluate import evaluate as evaluate_new_model

__all__ = [
    'NewModelConfig',
    'NewModel',
    'train_new_model',
    'evaluate_new_model'
]
