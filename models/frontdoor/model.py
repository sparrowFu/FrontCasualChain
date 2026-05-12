"""
FrontDoor Causal Chain 可训练模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FrontDoorConfig


class FrontDoorCausalModel(nn.Module):
    """
    可训练的前门准则因果链模型

    因果结构: 图像(I) → 共享语义(M) → 文本(T)
    """

    def __init__(
        self,
        image_feat_dim=2048,
        text_feat_dim=768,
        shared_dim=256,
        private_ratio=0.3,
        config=None
    ):
        """
        初始化模型

        Args:
            image_feat_dim: 图像特征维度（ResNet50输出）
            text_feat_dim: 文本特征维度（DistilBERT输出）
            shared_dim: 共享语义维度
            private_ratio: private特征占比
            config: 配置对象
        """
        super().__init__()

        self.config = config or FrontDoorConfig()
        self.shared_dim = shared_dim
        self.image_feat_dim = image_feat_dim
        self.text_feat_dim = text_feat_dim
        self.private_ratio = private_ratio

        # Shared/Private 分解器
        image_private_dim = int(image_feat_dim * private_ratio)
        self.image_shared_encoder = nn.Linear(image_feat_dim, shared_dim)
        self.image_private_encoder = nn.Linear(image_feat_dim, image_private_dim)

        text_private_dim = int(text_feat_dim * private_ratio)
        self.text_shared_encoder = nn.Linear(text_feat_dim, shared_dim)
        self.text_private_encoder = nn.Linear(text_feat_dim, text_private_dim)

        # 共享语义融合器
        self.semantic_fusion = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 因果效应估计器
        self.causal_effect_estimator = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(shared_dim // 2, 1)
        )

        # 重建解码器（用于重建损失）
        self.image_decoder = nn.Linear(shared_dim, image_feat_dim)
        self.text_decoder = nn.Linear(shared_dim, text_feat_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_to_shared_private(self, image_features, text_features):
        """
        将编码后的特征分解为 shared 和 private

        Args:
            image_features: [batch, image_feat_dim]
            text_features: [batch, text_feat_dim]

        Returns:
            dict: 包含 shared 和 private 特征
        """
        # 图像特征分解
        image_shared = self.image_shared_encoder(image_features)
        image_private = self.image_private_encoder(image_features)

        # 文本特征分解
        text_shared = self.text_shared_encoder(text_features)
        text_private = self.text_private_encoder(text_features)

        return {
            'image_shared': image_shared,
            'image_private': image_private,
            'text_shared': text_shared,
            'text_private': text_private
        }

    def compute_shared_semantic(self, image_shared, text_shared):
        """
        计算共享语义（中介变量 M）

        Args:
            image_shared: [batch, shared_dim]
            text_shared: [batch, shared_dim]

        Returns:
            shared_semantic: [batch, shared_dim]
        """
        # 拼接后融合
        concatenated = torch.cat([image_shared, text_shared], dim=-1)
        shared_semantic = self.semantic_fusion(concatenated)
        return shared_semantic

    def forward(self, image_features, text_features):
        """
        前向传播

        Args:
            image_features: [batch, image_feat_dim] 图像编码特征
            text_features: [batch, text_feat_dim] 文本编码特征

        Returns:
            dict: 包含各种中间输出和损失的字典
        """
        # 步骤1: Shared/Private 分解
        features = self.encode_to_shared_private(image_features, text_features)

        # 步骤2: 计算共享语义
        shared_semantic = self.compute_shared_semantic(
            features['image_shared'],
            features['text_shared']
        )

        # 步骤3: 计算因果效应
        causal_effect = self.causal_effect_estimator(shared_semantic)

        # 步骤4: 重建（用于重建损失）
        image_recon = self.image_decoder(features['image_shared'])
        text_recon = self.text_decoder(features['text_shared'])

        return {
            'image_shared': features['image_shared'],
            'image_private': features['image_private'],
            'text_shared': features['text_shared'],
            'text_private': features['text_private'],
            'shared_semantic': shared_semantic,
            'causal_effect': causal_effect,
            'image_recon': image_recon,
            'text_recon': text_recon
        }

    def get_causal_effect(self, image_features, text_features):
        """
        获取因果效应值（推理时使用）

        Args:
            image_features: [batch, image_feat_dim]
            text_features: [batch, text_feat_dim]

        Returns:
            causal_effect: [batch, 1]
        """
        with torch.no_grad():
            output = self.forward(image_features, text_features)
            return output['causal_effect']


class FrontDoorWithEncoders(nn.Module):
    """
    包含预训练编码器的完整 FrontDoor 模型
    """

    def __init__(self, image_encoder, text_encoder, causal_model):
        """
        Args:
            image_encoder: 预训练的图像编码器
            text_encoder: 预训练的文本编码器
            causal_model: FrontDoorCausalModel 实例
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.causal_model = causal_model

    def forward(self, batch):
        """
        完整前向传播：从原始输入到因果效应

        Args:
            batch: 包含 image, input_ids, attention_mask 的字典

        Returns:
            dict: 包含因果模型输出的字典
        """
        # 编码图像
        image_features = self.image_encoder(batch["image"])

        # 编码文本
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # 通过因果模型
        return self.causal_model(image_features, text_features)
