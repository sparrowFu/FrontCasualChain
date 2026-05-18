"""
VisionQA 测试脚本
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import argparse
from typing import Dict, List
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VisionQA.config import VQAConfig
from VisionQA.dataset import build_vqa_loaders
from VisionQA.model import CLIPVQAModel, FrontDoorVQAModel, load_clip_vqa_model, load_frontdoor_vqa_model


class VQAEvaluator:
    """VQA 任务评估器"""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        idx2answer: List[str],
        device: torch.device
    ):
        """
        Args:
            model: VQA 模型
            test_loader: 测试数据加载器
            idx2answer: 索引到答案的映射
            device: 设备
        """
        self.model = model
        self.test_loader = test_loader
        self.idx2answer = idx2answer
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        评估模型

        Returns:
            dict: 包含各种评估指标的字典
        """
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        # 按问题类型统计
        type_correct = [0] * 4
        type_total = [0] * 4

        # 预测结果
        predictions_list = []

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 前向传播
            output = self.model(batch)
            loss = output.get("loss", torch.tensor(0.0))

            # 预测
            pred_indices = torch.argmax(output["logits"], dim=-1)

            # 统计
            total_loss += loss.item()
            total_correct += (pred_indices == batch["answer_idx"]).sum().item()
            total_samples += batch["answer_idx"].size(0)

            # 按类型统计
            for i in range(len(batch["answer_idx"])):
                q_type = batch["question_type"][i].item()
                type_total[q_type] += 1
                if pred_indices[i] == batch["answer_idx"][i]:
                    type_correct[q_type] += 1

                # 保存预测结果
                predictions_list.append({
                    'question': batch['question'][i],
                    'ground_truth': batch['answer'][i],
                    'prediction': self.idx2answer[pred_indices[i].item()],
                    'type': q_type,
                    'correct': pred_indices[i].item() == batch["answer_idx"][i].item()
                })

        # 计算指标
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * total_correct / total_samples

        # 各类型准确率
        type_names = ['object', 'number', 'color', 'location']
        type_results = []
        for i in range(4):
            if type_total[i] > 0:
                acc = 100 * type_correct[i] / type_total[i]
                type_results.append({
                    'type': type_names[i],
                    'accuracy': acc,
                    'correct': type_correct[i],
                    'total': type_total[i]
                })
            else:
                type_results.append({
                    'type': type_names[i],
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0
                })

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': total_correct,
            'total': total_samples,
            'type_results': type_results,
            'predictions': predictions_list
        }

    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("VQA 测试结果")
        print("=" * 60)
        print(f"总体准确率: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print(f"平均损失: {results['loss']:.4f}")
        print("\n各问题类型准确率:")
        print("-" * 60)
        for tr in results['type_results']:
            print(f"{tr['type']:12s}: {tr['accuracy']:6.2f}% ({tr['correct']}/{tr['total']})")
        print("=" * 60)

    def show_examples(self, results: Dict, num_examples: int = 10):
        """显示预测示例"""
        print(f"\n预测示例 (前{num_examples}条):")
        print("-" * 80)

        correct_examples = [p for p in results['predictions'] if p['correct']]
        wrong_examples = [p for p in results['predictions'] if not p['correct']]

        type_names = ['object', 'number', 'color', 'location']

        for i, p in enumerate(results['predictions'][:num_examples]):
            status = "✓" if p['correct'] else "✗"
            print(f"{status} [{i+1}] 问题: {p['question']}")
            print(f"    类型: {type_names[p['type']]}, 真实: {p['ground_truth']}, 预测: {p['prediction']}")
            print()

    def save_results(self, results: Dict, save_path: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存简化结果（不包含完整预测列表）
        save_data = {
            'loss': results['loss'],
            'accuracy': results['accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'type_results': results['type_results']
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {save_path}")


def test_clip_vqa(args):
    """测试 CLIP VQA 模型"""
    config = VQAConfig.for_model('clip')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("测试 CLIP VQA 模型")
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
    idx2answer = train_dataset.idx2answer

    # 加载模型
    model_path = args.model_path or config.clip_vqa_save_path
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型或使用 --model-path 指定正确的路径")
        return

    print(f"加载模型: {model_path}")

    # 加载预训练的 CLIP 模型
    from models.clip.model import CLIPModel
    clip_checkpoint = torch.load(config.clip_model_path, map_location=device, weights_only=True)
    clip_model = CLIPModel()
    if isinstance(clip_checkpoint, dict) and 'model_state_dict' in clip_checkpoint:
        clip_model.load_state_dict(clip_checkpoint['model_state_dict'])
    else:
        clip_model.load_state_dict(clip_checkpoint)
    clip_model = clip_model.to(device)

    # 创建 VQA 模型
    model = CLIPVQAModel(
        clip_model=clip_model,
        num_answers=num_answers
    ).to(device)

    # 加载微调后的权重
    vqa_checkpoint = torch.load(model_path, map_location=device)
    if isinstance(vqa_checkpoint, dict) and 'model_state_dict' in vqa_checkpoint:
        model.load_state_dict(vqa_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(vqa_checkpoint)

    # 评估
    evaluator = VQAEvaluator(model, test_loader, idx2answer, device)
    results = evaluator.evaluate()

    # 打印结果
    evaluator.print_results(results)
    evaluator.show_examples(results, num_examples=args.examples)

    # 保存结果
    if args.save_path:
        evaluator.save_results(results, args.save_path)


def test_frontdoor_vqa(args):
    """测试 FrontDoor VQA 模型"""
    config = VQAConfig.for_model('frontdoor')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("测试 FrontDoor VQA 模型")
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
    idx2answer = train_dataset.idx2answer

    # 加载模型
    model_path = args.model_path or config.frontdoor_vqa_save_path
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型或使用 --model-path 指定正确的路径")
        return

    print(f"加载模型: {model_path}")

    # 加载预训练的 FrontDoor 模型
    from models.frontdoor.model import FrontDoorCausalModel, FrontDoorWithEncoders
    from models.clip.model import ImageEncoder, TextEncoder
    from models.frontdoor.config import FrontDoorConfig

    fd_config = FrontDoorConfig()
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()
    causal_model = FrontDoorCausalModel(
        image_feat_dim=fd_config.image_embedding,
        text_feat_dim=fd_config.text_embedding,
        shared_dim=fd_config.shared_dim,
        private_ratio=fd_config.private_ratio
    )
    frontdoor_model = FrontDoorWithEncoders(image_encoder, text_encoder, causal_model)

    fd_checkpoint = torch.load(config.frontdoor_model_path, map_location=device, weights_only=True)
    if isinstance(fd_checkpoint, dict) and 'model_state_dict' in fd_checkpoint:
        frontdoor_model.load_state_dict(fd_checkpoint['model_state_dict'])
    else:
        frontdoor_model.load_state_dict(fd_checkpoint)
    frontdoor_model = frontdoor_model.to(device)

    # 创建 VQA 模型
    model = FrontDoorVQAModel(
        frontdoor_model=frontdoor_model,
        num_answers=num_answers
    ).to(device)

    # 加载微调后的权重
    vqa_checkpoint = torch.load(model_path, map_location=device)
    if isinstance(vqa_checkpoint, dict) and 'model_state_dict' in vqa_checkpoint:
        model.load_state_dict(vqa_checkpoint['model_state_dict'])
    else:
        model.load_state_dict(vqa_checkpoint)

    # 评估
    evaluator = VQAEvaluator(model, test_loader, idx2answer, device)
    results = evaluator.evaluate()

    # 打印结果
    evaluator.print_results(results)
    evaluator.show_examples(results, num_examples=args.examples)

    # 保存结果
    if args.save_path:
        evaluator.save_results(results, args.save_path)


def main():
    parser = argparse.ArgumentParser(
        description='VisionQA 测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python VisionQA/test.py --model clip                          # 测试 CLIP VQA 模型
  python VisionQA/test.py --model frontdoor                     # 测试 FrontDoor VQA 模型
  python VisionQA/test.py --model clip --examples 20            # 显示更多示例
  python VisionQA/test.py --model clip --save-path results.json  # 保存结果
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
        '--model-path',
        type=str,
        default=None,
        help='模型文件路径（默认使用配置文件中的路径）'
    )

    parser.add_argument(
        '--examples',
        type=int,
        default=10,
        help='显示预测示例数量'
    )

    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='保存结果路径（JSON 格式）'
    )

    args = parser.parse_args()

    if args.model == 'clip':
        test_clip_vqa(args)
    elif args.model == 'frontdoor':
        test_frontdoor_vqa(args)
    else:
        print(f"未知模型: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
