"""
新模型配置
"""
from common.config import BaseConfig


class NewModelConfig(BaseConfig):
    """
    新模型的特定配置
    继承自BaseConfig，根据需要覆盖配置参数
    """

    # 模型特定配置
    model_name = 'your_model_name'
    model_path = "../models/your_model_path/"

    # 覆盖基础配置（如需要）
    # batch_size = 64
    # learning_rate = 1e-4

    @classmethod
    def to_dict(cls):
        """转换为字典格式"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
