"""
VisionQA 训练脚本
基于预训练 CLIP 和 FrontDoor 模型的 VQA 任务微调
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VisionQA.config import VQAConfig
from VisionQA.dataset import build_vqa_loaders
from VisionQA.model import CLIPVQAModel, FrontDoorVQAModel, load_clip_vqa_model, load_frontdoor_vqa_model


class VQATrainer:
    """VQA 任务训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: VQAConfig,
        device: torch.device
    ):
        """
        Args:
            model: VQA 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # 只优化 VQA 头的参数
        self.optimizer = torch.optim.AdamW(
            self.model.vqa_classifier.parameters(),
            lr=1e-4,
            weight_decay=1e-3
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

        self.best_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 前向传播
            output = self.model(batch)
            loss = output["loss"]

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(output["logits"], dim=-1)
            total_correct += (predictions == batch["answer_idx"]).sum().item()
            total_samples += batch["answer_idx"].size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * total_correct / total_samples:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * total_correct / total_samples

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # 按问题类型统计
        type_correct = [0] * 4
        type_total = [0] * 4

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 前向传播
            output = self.model(batch)
            loss = output["loss"]

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(output["logits"], dim=-1)
            total_correct += (predictions == batch["answer_idx"]).sum().item()
            total_samples += batch["answer_idx"].size(0)

            # 按类型统计
            for i in range(len(batch["answer_idx"])):
                q_type = batch["question_type"][i].item()
                type_total[q_type] += 1
                if predictions[i] == batch["answer_idx"][i]:
                    type_correct[q_type] += 1

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * total_correct / total_samples

        # 各类型准确率
        type_acc = []
        type_names = ['object', 'number', 'color', 'location']
        for i in range(4):
            if type_total[i] > 0:
                acc = 100 * type_correct[i] / type_total[i]
                type_acc.append((type_names[i], acc, type_total[i]))
            else:
                type_acc.append((type_names[i], 0, 0))

        return avg_loss, accuracy, type_acc

    def train(self, num_epochs: int):
        """完整训练流程"""
        print("=" * 60)
        print("开始 VQA 任务训练")
        print("=" * 60)
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"测试样本: {len(self.test_loader.dataset)}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {num_epochs}")
        print("=" * 60)

        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

            # 评估
            val_loss, val_acc, type_acc = self.evaluate()
            print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print("各类型准确率:")
            for name, acc, count in type_acc:
                print(f"  {name}: {acc:.2f}% (n={count})")

            # 学习率调度
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(self.config.vqa_save_path)
                print(f"保存最佳模型 (loss: {val_loss:.4f})")

        print("\n训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")

    def save_model(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)


def train_clip_vqa(args):
    """训练 CLIP VQA 模型"""
    config = VQAConfig.for_model('clip')

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("训练 CLIP VQA 模型")
    print("=" * 60)

    # 创建 tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)

    # 构建数据加载器
    train_loader, test_loader, train_dataset = build_vqa_loaders(
        train_dir=config.vqa_train_path,
        test_dir=config.vqa_test_path,
        tokenizer=tokenizer,
        image_path=config.images_path,
        config=config
    )

    num_answers = train_dataset.get_answer_vocab_size()
    print(f"答案词汇表大小: {num_answers}")

    # 加载预训练模型
    model = load_clip_vqa_model(
        clip_model_path=config.clip_model_path,
        num_answers=num_answers,
        device=device
    )

    # 创建训练器
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # 训练
    trainer.train(num_epochs=config.epochs)


def train_frontdoor_vqa(args):
    """训练 FrontDoor VQA 模型"""
    config = VQAConfig.for_model('frontdoor')

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("训练 FrontDoor VQA 模型")
    print("=" * 60)

    # 创建 tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)

    # 构建数据加载器
    train_loader, test_loader, train_dataset = build_vqa_loaders(
        train_dir=config.vqa_train_path,
        test_dir=config.vqa_test_path,
        tokenizer=tokenizer,
        image_path=config.images_path,
        config=config
    )

    num_answers = train_dataset.get_answer_vocab_size()
    print(f"答案词汇表大小: {num_answers}")

    # 加载预训练模型
    model = load_frontdoor_vqa_model(
        frontdoor_model_path=config.frontdoor_model_path,
        num_answers=num_answers,
        device=device
    )

    # 创建训练器
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # 训练
    trainer.train(num_epochs=config.epochs)


def main():
    parser = argparse.ArgumentParser(
        description='VisionQA 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python VisionQA/train.py --model clip              # 训练 CLIP VQA 模型
  python VisionQA/train.py --model frontdoor         # 训练 FrontDoor VQA 模型
  python VisionQA/train.py --model clip --epochs 10  # 自定义训练轮数
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='clip',
        choices=['clip', 'frontdoor'],
        help='选择基础模型 (clip/frontdoor)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批大小'
    )

    args = parser.parse_args()

    if args.model == 'clip':
        train_clip_vqa(args)
    elif args.model == 'frontdoor':
        train_frontdoor_vqa(args)
    else:
        print(f"未知模型: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
