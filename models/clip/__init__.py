"""
CLIP模型模块
"""
from .config import CLIPConfig
from .model import CLIPModel, ImageEncoder, TextEncoder, ProjectionHead
from .train import train as train_clip
from .evaluate import evaluate as evaluate_clip

__all__ = [
    'CLIPConfig',
    'CLIPModel',
    'ImageEncoder',
    'TextEncoder',
    'ProjectionHead',
    'train_clip',
    'evaluate_clip'
]
