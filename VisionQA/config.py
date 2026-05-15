"""
VisionQA 配置文件
"""
import torch
import os
from common.config import BaseConfig


class VQAConfig(BaseConfig):
    """VQA 任务配置"""

    def __init__(self):
        super().__init__()

        # ============ 数据集路径 ============
        self.vqa_train_path = os.path.join(self.dataset_path, 'train')
        self.vqa_test_path = os.path.join(self.dataset_path, 'test')
        self.images_path = os.path.join(self.dataset_path, 'images')

        # ============ 问题类型 ============
        # 0-object, 1-number, 2-color, 3-location
        self.num_types = 4
        self.type_names = ['object', 'number', 'color', 'location']

        # ============ 模型路径 ============
        self.clip_model_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\clipmodel\\best_model.pt"
        self.frontdoor_model_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\frontdoormodel\\best_model.pt"

        # ============ VQA 模型保存路径 ============
        self.vqa_results_dir = "D:\\code\\causality\\FrontdoorCausalChain\\results\\VisionQA"
        self.clip_vqa_save_path = os.path.join(self.vqa_results_dir, "clip_vqa_best_model.pt")
        self.frontdoor_vqa_save_path = os.path.join(self.vqa_results_dir, "frontdoor_vqa_best_model.pt")

        # ============ VQA 特定参数 ============
        # 答案词汇表大小（简单处理：使用预定义答案类别）
        self.num_answers = 1000  # 可根据实际数据集调整

        # 隐藏层维度
        self.vqa_hidden_dim = 512

        # Dropout
        self.vqa_dropout = 0.2

    @classmethod
    def for_model(cls, model_type: str):
        """
        根据模型类型创建配置

        Args:
            model_type: 'clip' 或 'frontdoor'
        """
        config = cls()
        if model_type == 'clip':
            config.model_path = config.clip_model_path
            config.vqa_save_path = config.clip_vqa_save_path
        elif model_type == 'frontdoor':
            config.model_path = config.frontdoor_model_path
            config.vqa_save_path = config.frontdoor_vqa_save_path
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        return config
