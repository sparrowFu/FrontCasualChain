"""
CLIP模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig
from .config import CLIPConfig


class ImageEncoder(nn.Module):
    """
    图像编码器 - 将图像编码为固定维度的向量
    """

    def __init__(self, model_name=None, pretrained=None, trainable=None):
        config = CLIPConfig()
        super().__init__()
        self.model = timm.create_model(
            model_name or config.model_name,
            pretrained,
            num_classes=0,
            global_pool="avg",
            pretrained_cfg_overlay=dict(file=config.image_model_path)
        )
        for p in self.model.parameters():
            p.requires_grad = trainable if trainable is not None else config.trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    """
    文本编码器 - 将文本编码为固定维度的向量
    """

    def __init__(self, model_name=None, pretrained=None, trainable=None):
        config = CLIPConfig()
        super().__init__()
        model_path = config.text_model_path

        if pretrained if pretrained is not None else config.pretrained:
            self.model = DistilBertModel.from_pretrained(model_path)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable if trainable is not None else config.trainable

        # 使用CLS token的隐藏表示作为句子嵌入
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """
    投影头 - 将编码器输出映射到共享的嵌入空间
    """

    def __init__(self, embedding_dim, projection_dim=None, dropout=None):
        config = CLIPConfig()
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim or config.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim or config.projection_dim, projection_dim or config.projection_dim)
        self.dropout = nn.Dropout(dropout or config.dropout)
        self.layer_norm = nn.LayerNorm(projection_dim or config.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    """
    CLIP主模型 - 实现图文对比学习
    """

    def __init__(self, temperature=None, image_embedding=None, text_embedding=None):
        config = CLIPConfig()
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding or config.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding or config.text_embedding)
        self.temperature = temperature if temperature is not None else config.temperature

    def forward(self, batch):
        # 获取图像和文本特征
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # 获取图像和文本嵌入（相同维度）
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # 归一化嵌入以匹配余弦相似度目标
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # 计算相似度logits
        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        # 构建目标标签
        ids = batch["id"]
        if ids.ndim > 1:
            ids = ids.view(ids.size(0))
        positive_mask = ids.unsqueeze(1) == ids.unsqueeze(0)
        positive_counts = positive_mask.sum(dim=-1, keepdim=True)
        targets = positive_mask.float() / positive_counts.clamp_min(1.0)

        # 计算损失
        texts_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        """
        自定义交叉熵损失函数
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
