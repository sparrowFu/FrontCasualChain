"""
CLIP模型训练脚本
"""
import os
import torch
import itertools
from pathlib import Path
from transformers import DistilBertTokenizer
from .config import CLIPConfig
from .model import CLIPModel
from common import load_mscoco_data, train_epoch, valid_epoch, build_loaders


def train(model_path="best.pt", resume=False):
    """
    训练CLIP模型

    Args:
        model_path: 模型保存路径
        resume: 是否从checkpoint恢复
    """
    config = CLIPConfig()

    print("准备数据...")
    train_df, valid_df = load_mscoco_data(config=config, test_size=0.2, random_state=42)
    tokenizer_path = Path(config.text_model_path)

    tokenizer = DistilBertTokenizer.from_pretrained(
        config.text_model_path,
        local_files_only=True
    )
    train_loader = build_loaders(train_df, tokenizer, mode="train", config=config)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid", config=config)

    print("初始化模型...")
    model = CLIPModel().to(config.device)

    print(f"选择的设备：{config.device}")

    if resume and os.path.exists(model_path):
        print(f"从 {model_path} 恢复模型...")
        model.load_state_dict(torch.load(model_path, map_location=config.device))

    # 设置不同模块的学习率
    params = [
        {"params": model.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": config.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(),
            model.text_projection.parameters()
        ), "lr": config.head_lr, "weight_decay": config.weight_decay}
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.factor
    )
    step = "epoch"

    print("开始训练...")
    best_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch + 1}/{config.epochs}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        # 保存最佳模型
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), config.model_save_path)
            print(f"✅ 保存最佳模型! 验证损失: {best_loss:.4f}")

        lr_scheduler.step(valid_loss.avg)

    print(f"\n训练完成! 最佳验证损失: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
