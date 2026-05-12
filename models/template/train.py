"""
新模型训练脚本
"""
import torch
from transformers import DistilBertTokenizer
from .config import NewModelConfig
from .model import NewModel
from common import make_train_valid_dfs, build_loaders, train_epoch, valid_epoch


def train(model_path="best_new_model.pt"):
    """
    训练新模型

    Args:
        model_path: 模型保存路径
    """
    config = NewModelConfig()

    print("准备数据...")
    train_df, valid_df = make_train_valid_dfs(test_size=0.2, random_state=42)
    tokenizer = DistilBertTokenizer.from_pretrained(config.model_path)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    print("初始化模型...")
    model = NewModel(config).to(config.device)

    # 设置优化器
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 设置学习率调度器
    # lr_scheduler = ...

    print("开始训练...")
    best_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch + 1}/{config.epochs}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step="epoch")

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        # 保存最佳模型
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), model_path)
            print(f"✅ 保存最佳模型! 验证损失: {best_loss:.4f}")

    print(f"\n训练完成! 最佳验证损失: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
