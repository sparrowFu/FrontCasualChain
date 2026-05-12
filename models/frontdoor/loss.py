"""
FrontDoor Causal Chain 损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FrontDoorConfig


class FrontDoorLoss(nn.Module):
    """
    FrontDoor 因果链的组合损失函数

    包含:
    1. Shared 特征对齐损失 - 让匹配对的 shared 特征相似
    2. 正交损失 - 确保 shared 和 private 特征正交
    3. 对比学习损失 - 区分匹配和不匹配对
    4. 重建损失 - 确保信息保留
    """

    def __init__(self, config=None):
        """
        Args:
            config: FrontDoorConfig 配置对象
        """
        super().__init__()
        self.config = config or FrontDoorConfig()

        # 损失权重
        self.lambda_alignment = self.config.lambda_alignment
        self.lambda_orthogonal = self.config.lambda_orthogonal
        self.lambda_contrastive = self.config.lambda_contrastive
        self.lambda_reconstruction = self.config.lambda_reconstruction

        self.temperature = self.config.temperature

    def alignment_loss(self, image_shared, text_shared):
        """
        Shared 特征对齐损失

        目标: 让匹配对的 shared 特征尽可能相似

        Args:
            image_shared: [batch, shared_dim]
            text_shared: [batch, shared_dim]

        Returns:
            loss: 标量
        """
        # 使用余弦相似度
        similarity = F.cosine_similarity(image_shared, text_shared, dim=-1)
        # 最大化相似度 = 最小化 (1 - 相似度)
        loss = (1 - similarity).mean()
        return loss

    def orthogonal_loss(self, features):
        """
        正交损失

        目标: 确保 shared 和 private 特征正交（无信息重叠）

        Args:
            features: 包含 shared 和 private 特征的字典

        Returns:
            loss: 标量
        """
        img_shared = features['image_shared']
        img_private = features['image_private']
        txt_shared = features['text_shared']
        txt_private = features['text_private']

        # 计算相关系数作为正交性度量
        def correlation(x, y):
            x_mean = x.mean(dim=-1, keepdim=True)
            y_mean = y.mean(dim=-1, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean

            # 填充到较大维度
            max_dim = max(x.size(-1), y.size(-1))
            if x.size(-1) < max_dim:
                padding = max_dim - x.size(-1)
                x_centered = F.pad(x_centered, (0, padding))
            if y.size(-1) < max_dim:
                padding = max_dim - y.size(-1)
                y_centered = F.pad(y_centered, (0, padding))

            cov = (x_centered * y_centered).mean(dim=-1)
            x_std = x_centered.std(dim=-1)
            y_std = y_centered.std(dim=-1)
            return (cov / (x_std * y_std + 1e-8)).abs().mean()

        # 图像的正交损失
        img_ortho_loss = correlation(img_shared, img_private)
        # 文本的正交损失
        txt_ortho_loss = correlation(txt_shared, txt_private)

        return (img_ortho_loss + txt_ortho_loss) / 2

    def contrastive_loss(self, shared_semantic, ids):
        """
        对比学习损失

        目标: 同一样本的 shared semantic 应该相似，不同样本的应该不相似

        Args:
            shared_semantic: [batch, shared_dim]
            ids: [batch] 样本ID（同一样本的不同文本描述有相同ID）

        Returns:
            loss: 标量
        """
        batch_size = shared_semantic.size(0)

        # 归一化
        shared_semantic = F.normalize(shared_semantic, p=2, dim=-1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(shared_semantic, shared_semantic.T)

        # 创建标签矩阵（同一样本为正样本）
        if ids.ndim > 1:
            ids = ids.view(ids.size(0))
        labels = ids.unsqueeze(1) == ids.unsqueeze(0)
        labels = labels.float()

        # 温度缩放
        similarity_matrix = similarity_matrix / self.temperature

        # 交叉熵损失
        log_prob = F.log_softmax(similarity_matrix, dim=-1)
        loss = -(labels * log_prob).sum(dim=-1).mean()

        return loss

    def reconstruction_loss(self, image_recon, text_recon, image_features, text_features):
        """
        重建损失

        目标: 确保从 shared 特征重建时保留原始信息

        Args:
            image_recon: 重建的图像特征
            text_recon: 重建的文本特征
            image_features: 原始图像特征
            text_features: 原始文本特征

        Returns:
            loss: 标量
        """
        img_loss = F.mse_loss(image_recon, image_features)
        txt_loss = F.mse_loss(text_recon, text_features)

        # 归一化（不同特征量级可能不同）
        img_loss = img_loss / (image_features.var() + 1e-8)
        txt_loss = txt_loss / (text_features.var() + 1e-8)

        return (img_loss + txt_loss) / 2

    def forward(self, model_output, image_features, text_features, ids):
        """
        计算总损失

        Args:
            model_output: FrontDoorCausalModel 的输出字典
            image_features: 原始图像特征
            text_features: 原始文本特征
            ids: 样本ID

        Returns:
            dict: 包含总损失和各项子损失的字典
        """
        # 1. Shared 特征对齐损失
        align_loss = self.alignment_loss(
            model_output['image_shared'],
            model_output['text_shared']
        )

        # 2. 正交损失
        ortho_loss = self.orthogonal_loss(model_output)

        # 3. 对比学习损失
        contrast_loss = self.contrastive_loss(
            model_output['shared_semantic'],
            ids
        )

        # 4. 重建损失
        recon_loss = self.reconstruction_loss(
            model_output['image_recon'],
            model_output['text_recon'],
            image_features,
            text_features
        )

        # 总损失
        total_loss = (
            self.lambda_alignment * align_loss +
            self.lambda_orthogonal * ortho_loss +
            self.lambda_contrastive * contrast_loss +
            self.lambda_reconstruction * recon_loss
        )

        return {
            'total_loss': total_loss,
            'alignment_loss': align_loss,
            'orthogonal_loss': ortho_loss,
            'contrastive_loss': contrast_loss,
            'reconstruction_loss': recon_loss
        }

    def get_metrics(self, model_output):
        """
        获取训练指标（用于监控）

        Args:
            model_output: 模型输出字典

        Returns:
            dict: 各种指标
        """
        # Shared 特征相似度
        shared_similarity = F.cosine_similarity(
            model_output['image_shared'],
            model_output['text_shared'],
            dim=-1
        ).mean().item()

        # Shared-Private 正交性（相关系数）
        def correlation(x, y):
            x_mean = x.mean(dim=-1, keepdim=True)
            y_mean = y.mean(dim=-1, keepdim=True)
            x_centered = x - x_mean
            y_centered = y - y_mean

            # 填充到较大维度
            max_dim = max(x.size(-1), y.size(-1))
            if x.size(-1) < max_dim:
                padding = max_dim - x.size(-1)
                x_centered = F.pad(x_centered, (0, padding))
            if y.size(-1) < max_dim:
                padding = max_dim - y.size(-1)
                y_centered = F.pad(y_centered, (0, padding))

            cov = (x_centered * y_centered).mean(dim=-1)
            x_std = x_centered.std(dim=-1)
            y_std = y_centered.std(dim=-1)
            return (cov / (x_std * y_std + 1e-8)).abs().mean()

        img_ortho = correlation(
            model_output['image_shared'],
            model_output['image_private']
        )
        txt_ortho = correlation(
            model_output['text_shared'],
            model_output['text_private']
        )

        return {
            'shared_similarity': shared_similarity,
            'image_orthogonality': img_ortho.item(),
            'text_orthogonality': txt_ortho.item(),
            'avg_orthogonality': ((img_ortho + txt_ortho) / 2).item()
        }
