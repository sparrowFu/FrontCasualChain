"""
CLIP模型配置
"""
import torch
from common.config import BaseConfig
from pathlib import Path


class CLIPConfig(BaseConfig):
    """CLIP模型特定配置"""

    def __init__(self):
        super().__init__()
        # 模型名称
        self.model_name = 'resnet50'
        self.text_encoder_model = "distilbert-base-uncased"

        # 模型保存路径
        self.model_save_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\clipmodel\\best_model.pt"
        self.checkpoint_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\clipmodel\\checkpoint.pt"

        # 投影头参数
        self.num_projection_layers = 1

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
