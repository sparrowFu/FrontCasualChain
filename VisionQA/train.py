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
        device: torch.device,
        resume_path: str = None
    ):
        """
        Args:
            model: VQA 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            config: 配置对象
            device: 设备
            resume_path: 恢复训练的检查点路径
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
        self.start_epoch = 1

        # 如果指定了恢复路径，加载检查点
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

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
        print(f"训练轮数: {self.start_epoch}-{num_epochs}")
        print("=" * 60)

        for epoch in range(self.start_epoch, num_epochs + 1):
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
                self.save_model(self.config.vqa_save_path, epoch=epoch)
                print(f"保存最佳模型 (loss: {val_loss:.4f})")

        print("\n训练完成!")
        print(f"最佳验证损失: {self.best_loss:.4f}")

    def save_model(self, path: str, epoch: int = None):
        """
        保存模型检查点

        Args:
            path: 保存路径
            epoch: 当前训练轮数
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)

    def load_checkpoint(self, path: str):
        """
        从检查点恢复训练

        Args:
            path: 检查点路径
        """
        print(f"从检查点恢复训练: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 加载模型状态
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复训练状态
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"从第 {checkpoint['epoch']} 轮继续训练")
        else:
            self.start_epoch = 1

        print(f"当前最佳损失: {self.best_loss:.4f}")


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
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_path)

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

    # 确定加载路径
    if args.resume:
        # 从已保存的 VQA 模型继续训练 - 直接加载完整模型
        print(f"从已保存的模型继续训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # 如果 resume 时没有指定 epochs，默认再训练 5 个 epoch
        if not args.epochs:
            config.epochs = 5
            print(f"恢复训练：将再训练 {config.epochs} 个 epoch")
            print(f"提示：使用 --epochs 参数可以指定训练轮数")

        # 首先需要加载预训练的 CLIP 模型
        model = load_clip_vqa_model(
            clip_model_path=config.clip_model_path,
            num_answers=num_answers,
            device=device
        )
        # 然后加载完整的 VQA 模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载完整 VQA 模型权重")
    else:
        # 从预训练模型开始
        model = load_clip_vqa_model(
            clip_model_path=config.clip_model_path,
            num_answers=num_answers,
            device=device
        )

    # 创建训练器
    resume_path = args.resume if args.resume else None
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        resume_path=resume_path
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
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_model_path)

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

    # 确定加载路径
    if args.resume:
        # 从已保存的 VQA 模型继续训练 - 直接加载完整模型
        print(f"从已保存的模型继续训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        # 如果 resume 时没有指定 epochs，默认再训练 5 个 epoch
        if not args.epochs:
            config.epochs = 5
            print(f"恢复训练：将再训练 {config.epochs} 个 epoch")
            print(f"提示：使用 --epochs 参数可以指定训练轮数")

        # 首先需要加载预训练的 FrontDoor 模型
        model = load_frontdoor_vqa_model(
            frontdoor_model_path=config.frontdoor_model_path,
            num_answers=num_answers,
            device=device
        )
        # 然后加载完整的 VQA 模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载完整 VQA 模型权重")
    else:
        # 从预训练模型开始
        model = load_frontdoor_vqa_model(
            frontdoor_model_path=config.frontdoor_model_path,
            num_answers=num_answers,
            device=device
        )

    # 创建训练器
    resume_path = args.resume if args.resume else None
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        resume_path=resume_path
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
  python VisionQA/train.py --model clip --resume D:\\code\\causality\\FrontdoorCausalChain\\results\\VisionQA\\clip_vqa_best_model.pt              # 从检查点继续训练（默认再训练 5 个 epoch）
  python VisionQA/train.py --model clip --resume D:\\code\\causality\\FrontdoorCausalChain\\results\\VisionQA\\clip_vqa_best_model.pt --epochs 10  # 从检查点继续训练并指定 10 个 epoch
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

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点继续训练的模型路径'
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
