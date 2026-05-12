"""
FrontDoor Causal Chain 评估脚本
"""
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DistilBertTokenizer

from .config import FrontDoorConfig
from .model import FrontDoorCausalModel, FrontDoorWithEncoders
from common.data import make_train_valid_dfs
from models.clip.model import ImageEncoder, TextEncoder


def load_model_for_eval(model_path, config, device):
    """
    加载训练好的模型用于评估

    Args:
        model_path: 模型权重路径
        config: 配置对象
        device: torch设备

    Returns:
        model: FrontDoorWithEncoders 模型
    """
    # 加载预训练编码器
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # 设置为评估模式并冻结
    image_encoder.eval()
    text_encoder.eval()
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # 创建因果模型
    causal_model = FrontDoorCausalModel(
        image_feat_dim=config.image_embedding,
        text_feat_dim=config.text_embedding,
        shared_dim=config.shared_dim,
        private_ratio=config.private_ratio,
        config=config
    ).to(device)

    # 加载训练好的权重
    if os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        causal_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用未训练的模型进行评估")

    causal_model.eval()

    # 组合模型
    model = FrontDoorWithEncoders(image_encoder, text_encoder, causal_model)
    return model


def evaluate_single_pair(model, image, text_input_ids, text_attention_mask, device):
    """
    评估单个图像-文本对

    Args:
        model: FrontDoorWithEncoders 模型
        image: 图像张量
        text_input_ids: 文本输入ID
        text_attention_mask: 文本注意力掩码
        device: torch设备

    Returns:
        dict: 包含因果效应和各种指标的字典
    """
    model.eval()

    with torch.no_grad():
        batch = {
            "image": image.unsqueeze(0).to(device),
            "input_ids": text_input_ids.unsqueeze(0).to(device),
            "attention_mask": text_attention_mask.unsqueeze(0).to(device)
        }

        output = model(batch)

        # 获取编码特征
        image_features = model.image_encoder(batch["image"])
        text_features = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # 计算指标
        shared_similarity = torch.nn.functional.cosine_similarity(
            output['image_shared'],
            output['text_shared'],
            dim=-1
        ).item()

        return {
            'causal_effect': output['causal_effect'].item(),
            'shared_similarity': shared_similarity,
            'shared_semantic': output['shared_semantic'].cpu(),
            'image_shared': output['image_shared'].cpu(),
            'text_shared': output['text_shared'].cpu()
        }


def visualize_causal_chain(results, save_path='frontdoor_evaluation.png'):
    """
    可视化评估结果

    Args:
        results: 评估结果列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FrontDoor 因果链评估结果', fontsize=16, fontweight='bold')

    # 1. 因果效应分布
    ax = axes[0, 0]
    causal_effects = [r['causal_effect'] for r in results]
    ax.hist(causal_effects, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('因果效应分布', fontweight='bold')
    ax.set_xlabel('因果效应值')
    ax.set_ylabel('频次')
    ax.grid(True, alpha=0.3)

    # 2. Shared相似度分布
    ax = axes[0, 1]
    shared_similarities = [r['shared_similarity'] for r in results]
    ax.hist(shared_similarities, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='阈值(0.5)')
    ax.set_title('Shared特征相似度分布', fontweight='bold')
    ax.set_xlabel('余弦相似度')
    ax.set_ylabel('频次')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 因果效应 vs Shared相似度
    ax = axes[1, 0]
    ax.scatter(shared_similarities, causal_effects, alpha=0.5)
    ax.set_title('因果效应 vs Shared相似度', fontweight='bold')
    ax.set_xlabel('Shared相似度')
    ax.set_ylabel('因果效应')
    ax.grid(True, alpha=0.3)

    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    统计摘要

    样本数: {len(results)}

    因果效应:
      均值: {sum(causal_effects)/len(causal_effects):.4f}
      最小值: {min(causal_effects):.4f}
      最大值: {max(causal_effects):.4f}

    Shared相似度:
      均值: {sum(shared_similarities)/len(shared_similarities):.4f}
      最小值: {min(shared_similarities):.4f}
      最大值: {max(shared_similarities):.4f}

    前门准则验证:
      {'✅ 满足' if sum(shared_similarities)/len(shared_similarities) > 0.5 else '❌ 不满足'}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存至: {save_path}")
    plt.close()


def evaluate(config=None, num_samples=100):
    """
    评估 FrontDoor 因果链模型

    Args:
        config: FrontDoorConfig 配置对象
        num_samples: 评估样本数量
    """
    config = config or FrontDoorConfig()
    device = config.device

    print("=" * 60)
    print("FrontDoor 因果链评估")
    print("=" * 60)

    # 准备数据
    print("\n准备数据...")
    _, valid_df = make_train_valid_dfs(test_size=0.2, random_state=42)
    valid_df = valid_df.head(num_samples)
    print(f"评估样本数: {len(valid_df)}")

    # 加载tokenizer
    model_path = str(config.text_model_path).rstrip(os.sep)
    tokenizer = DistilBertTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    # 加载模型
    print("\n加载模型...")
    model = load_model_for_eval(config.model_save_path, config, device)

    # 评估
    print("\n开始评估...")
    results = []

    from FrontdoorCausalChain.common.BaseDataset import get_transforms
    from PIL import Image as PILImage

    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
        # 加载图像
        image_path = os.path.join(config.image_path, row['image'])
        try:
            image = PILImage.open(image_path).convert('RGB')
            image = get_transforms()({'image': image})['image']
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            continue

        # 编码文本
        text_encoding = tokenizer(
            row['caption'],
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 评估
        result = evaluate_single_pair(
            model,
            image,
            text_encoding['input_ids'].squeeze(0),
            text_encoding['attention_mask'].squeeze(0),
            device
        )
        results.append(result)

    # 统计结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    causal_effects = [r['causal_effect'] for r in results]
    shared_similarities = [r['shared_similarity'] for r in results]

    print(f"\n样本数: {len(results)}")
    print(f"\n因果效应:")
    print(f"  均值: {sum(causal_effects)/len(causal_effects):.4f}")
    print(f"  最小值: {min(causal_effects):.4f}")
    print(f"  最大值: {max(causal_effects):.4f}")

    print(f"\nShared相似度:")
    print(f"  均值: {sum(shared_similarities)/len(shared_similarities):.4f}")
    print(f"  最小值: {min(shared_similarities):.4f}")
    print(f"  最大值: {max(shared_similarities):.4f}")

    print(f"\n前门准则验证:")
    avg_similarity = sum(shared_similarities) / len(shared_similarities)
    if avg_similarity > 0.5:
        print(f"  ✅ 满足 (平均相似度: {avg_similarity:.4f} > 0.5)")
    else:
        print(f"  ❌ 不满足 (平均相似度: {avg_similarity:.4f} < 0.5)")

    # 可视化
    print("\n生成可视化...")
    visualize_causal_chain(results)

    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # 创建配置
    config = FrontDoorConfig()

    # 评估模型
    results = evaluate(config, num_samples=100)
