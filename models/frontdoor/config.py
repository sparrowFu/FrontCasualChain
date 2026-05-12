"""
FrontDoor Causal Chain 模型配置
"""
import torch
from common.config import BaseConfig


class FrontDoorConfig(BaseConfig):
    """FrontDoor 因果链训练配置"""

    def __init__(self):
        """初始化配置"""
        super().__init__()
        # 模型名称
        self.model_name = 'frontdoor'

        # 模型架构参数
        self.shared_dim = 256              # 共享语义维度
        self.private_ratio = 0.3           # private特征比例

        # 损失函数权重
        self.lambda_alignment = 1.0        # shared特征对齐损失权重
        self.lambda_orthogonal = 0.1       # 正交损失权重
        self.lambda_contrastive = 1.0      # 对比学习损失权重
        self.lambda_reconstruction = 0.5   # 重建损失权重

        # 训练参数
        self.epochs = 2
        self.batch_size = 32
        self.lr = 1e-5
        self.weight_decay = 1e-4

        # 温度参数（用于对比学习）
        self.temperature = 0.07

        # 保存路径
        self.model_save_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\frontdoormodel\\best_model.pt"
        self.checkpoint_path = "D:\\code\\causality\\FrontdoorCausalChain\\results\\frontdoormodel\\checkpoint.pt"

        # 日志
        self.log_interval = 10             # 每10个batch打印一次
        self.save_interval = 1             # 每1个epoch保存一次

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
