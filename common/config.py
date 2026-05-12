"""
基础配置类
"""
import torch
import os


class BaseConfig:
    """所有模型的基类配置"""

    # ============ 项目根目录 ============
    project_root = "D:\\code\\causality\\FrontdoorCausalChain"

    def __init__(self):
        """初始化配置"""
        # ============ 数据集配置 ============
        # 使用 MSCOCO Captions 数据集
        self.dataset_path = os.path.join(self.project_root, 'data', 'mscoco_captions')
        self.captions_path = os.path.join(self.dataset_path, 'captions')
        self.images_path = os.path.join(self.dataset_path, 'images')

        # ============ 文本模型路径 ============
        self.text_model_path = os.path.join(self.project_root, 'PreTrainedModels', 'distilbert_base_uncased')

        # ============ 图像模型路径 ============
        self.image_model_path = os.path.join(self.project_root, 'PreTrainedModels', 'resnet50', 'pytorch_model.bin')

    # ============ 训练参数 ============
    batch_size = 32
    num_workers = 4
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 优化器参数 ============
    head_lr = 2 * 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8

    # ============ 图像参数 ============
    size = 224
    image_embedding = 2048

    # ============ 文本参数 ============
    text_embedding = 768
    max_length = 200
    text_tokenizer = "distilbert-base-uncased"

    # ============ 通用参数 ============
    pretrained = True
    trainable = True
    temperature = 1.0
    projection_dim = 256
    dropout = 0.1

    # ============ 调试模式 ============
    debug = False
