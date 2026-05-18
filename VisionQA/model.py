"""
VisionQA 模型定义
基于预训练的 CLIP 和 FrontDoor 模型进行 VQA 任务微调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .config import VQAConfig


class VQAClassifier(nn.Module):
    """
    VQA 分类器头
    将图文融合特征映射到答案空间
    """

    def __init__(
        self,
        input_dim: int,
        num_answers: int,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: 输入特征维度
            num_answers: 答案类别数量
            hidden_dim: 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_answers)
        )

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

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] 融合特征

        Returns:
            logits: [batch, num_answers]
        """
        return self.classifier(x)


class CLIPVQAModel(nn.Module):
    """
    基于 CLIP 的 VQA 模型
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_answers: int,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Args:
            clip_model: 预训练的 CLIP 模型
            num_answers: 答案类别数量
            hidden_dim: VQA 头隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        # 冻结 CLIP 模型的编码器部分
        self.image_encoder = clip_model.image_encoder
        self.text_encoder = clip_model.text_encoder

        # 冻结编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 获取投影头
        self.image_projection = clip_model.image_projection
        self.text_projection = clip_model.text_projection

        # 冻结投影头（可选）
        # for param in self.image_projection.parameters():
        #     param.requires_grad = False
        # for param in self.text_projection.parameters():
        #     param.requires_grad = False

        # 计算输入维度（两个投影输出的拼接）
        projection_dim = clip_model.image_projection.projection.out_features

        # VQA 分类器
        self.vqa_classifier = VQAClassifier(
            input_dim=projection_dim * 2,
            num_answers=num_answers,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, batch):
        """
        Args:
            batch: 包含 image, input_ids, attention_mask 的字典

        Returns:
            dict: {
                'logits': [batch, num_answers],
                'loss': scalar (需要提供 answer_idx)
            }
        """
        # 获取图像特征
        image_features = self.image_encoder(batch["image"])

        # 获取文本特征
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # 获取投影后的嵌入
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # 拼接图文嵌入
        fused = torch.cat([image_embeddings, text_embeddings], dim=-1)

        # 分类
        logits = self.vqa_classifier(fused)

        output = {"logits": logits}

        # 计算损失（如果提供了答案）
        if "answer_idx" in batch:
            loss = F.cross_entropy(logits, batch["answer_idx"])
            output["loss"] = loss

        return output

    def predict(self, batch):
        """
        预测答案

        Args:
            batch: 输入批次

        Returns:
            predictions: [batch] 预测的答案索引
        """
        with torch.no_grad():
            output = self.forward(batch)
            predictions = torch.argmax(output["logits"], dim=-1)
        return predictions


class FrontDoorVQAModel(nn.Module):
    """
    基于 FrontDoor 因果链模型的 VQA 模型
    """

    def __init__(
        self,
        frontdoor_model: nn.Module,
        num_answers: int,
        hidden_dim: int = 512,
        dropout: float = 0.2
    ):
        """
        Args:
            frontdoor_model: 预训练的 FrontDoorWithEncoders 模型
            num_answers: 答案类别数量
            hidden_dim: VQA 头隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        # 冻结 FrontDoor 模型
        self.frontdoor_model = frontdoor_model
        for param in self.frontdoor_model.parameters():
            param.requires_grad = False

        # 共享语义维度
        self.shared_dim = frontdoor_model.causal_model.shared_dim

        # VQA 分类器（使用共享语义作为输入）
        self.vqa_classifier = VQAClassifier(
            input_dim=self.shared_dim,
            num_answers=num_answers,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, batch):
        """
        Args:
            batch: 包含 image, input_ids, attention_mask 的字典

        Returns:
            dict: {
                'logits': [batch, num_answers],
                'loss': scalar (需要提供 answer_idx)
            }
        """
        # 获取 FrontDoor 模型输出
        with torch.set_grad_enabled(False):
            frontdoor_output = self.frontdoor_model(batch)
            shared_semantic = frontdoor_output['shared_semantic']

        # 分类
        logits = self.vqa_classifier(shared_semantic)

        output = {"logits": logits}

        # 计算损失（如果提供了答案）
        if "answer_idx" in batch:
            loss = F.cross_entropy(logits, batch["answer_idx"])
            output["loss"] = loss

        return output

    def predict(self, batch):
        """
        预测答案

        Args:
            batch: 输入批次

        Returns:
            predictions: [batch] 预测的答案索引
        """
        with torch.no_grad():
            output = self.forward(batch)
            predictions = torch.argmax(output["logits"], dim=-1)
        return predictions


def load_clip_vqa_model(
    clip_model_path: str,
    num_answers: int,
    device: torch.device,
    hidden_dim: int = 512,
    dropout: float = 0.2
) -> CLIPVQAModel:
    """
    加载 CLIP VQA 模型

    Args:
        clip_model_path: CLIP 模型路径
        num_answers: 答案类别数量
        device: 设备
        hidden_dim: VQA 头隐藏层维度
        dropout: Dropout 比例

    Returns:
        CLIPVQAModel 实例
    """
    # 加载预训练的 CLIP 模型
    checkpoint = torch.load(clip_model_path, map_location=device, weights_only=True)

    # 重建 CLIP 模型
    from models.clip.model import CLIPModel
    clip_model = CLIPModel()

    # 加载权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        clip_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        clip_model.load_state_dict(checkpoint)

    clip_model = clip_model.to(device)

    # 创建 VQA 模型
    vqa_model = CLIPVQAModel(
        clip_model=clip_model,
        num_answers=num_answers,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    return vqa_model


def load_frontdoor_vqa_model(
    frontdoor_model_path: str,
    num_answers: int,
    device: torch.device,
    hidden_dim: int = 512,
    dropout: float = 0.2
) -> FrontDoorVQAModel:
    """
    加载 FrontDoor VQA 模型

    Args:
        frontdoor_model_path: FrontDoor 模型路径
        num_answers: 答案类别数量
        device: 设备
        hidden_dim: VQA 头隐藏层维度
        dropout: Dropout 比例

    Returns:
        FrontDoorVQAModel 实例
    """
    # 加载预训练的 FrontDoor 模型
    checkpoint = torch.load(frontdoor_model_path, map_location=device, weights_only=False)

    # 重建 FrontDoor 模型
    from models.frontdoor.model import FrontDoorCausalModel, FrontDoorWithEncoders
    from models.clip.model import ImageEncoder, TextEncoder
    from models.frontdoor.config import FrontDoorConfig

    config = FrontDoorConfig()

    # 创建编码器
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()

    # 创建因果模型
    causal_model = FrontDoorCausalModel(
        image_feat_dim=config.image_embedding,
        text_feat_dim=config.text_embedding,
        shared_dim=config.shared_dim,
        private_ratio=config.private_ratio
    )

    # 创建完整模型
    frontdoor_model = FrontDoorWithEncoders(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        causal_model=causal_model
    )

    # 加载权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # 如果 state_dict 的键没有 'causal_model.' 前缀，添加它
    # 这是因为训练时只保存了 causal_model.state_dict()
    if not any(k.startswith('causal_model.') for k in state_dict.keys()):
        state_dict = {f'causal_model.{k}': v for k, v in state_dict.items()}

    # 使用 strict=False 因为 checkpoint 可能不包含 image_encoder 和 text_encoder 的权重
    frontdoor_model.load_state_dict(state_dict, strict=False)

    frontdoor_model = frontdoor_model.to(device)

    # 创建 VQA 模型
    vqa_model = FrontDoorVQAModel(
        frontdoor_model=frontdoor_model,
        num_answers=num_answers,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    return vqa_model
