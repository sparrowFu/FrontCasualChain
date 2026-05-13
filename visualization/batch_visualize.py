"""
批量因果链可视化

对多张图片进行批量因果推断和可视化
"""
import os
import sys
import argparse
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from visualization.causal_visualizer import CausalChainVisualizer


def batch_visualize(image_dir: str,
                    texts: list,
                    output_dir: str,
                    clip_model: str = None,
                    frontdoor_model: str = None,
                    device: str = None):
    """
    批量可视化

    Args:
        image_dir: 图片目录
        texts: 对应的文本描述列表
        output_dir: 输出目录
        clip_model: CLIP 模型路径
        frontdoor_model: FrontDoor 模型路径
        device: 设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化可视化器
    print("初始化因果链可视化器...")
    visualizer = CausalChainVisualizer(
        clip_model_path=clip_model,
        frontdoor_model_path=frontdoor_model,
        device=device
    )

    # 获取所有图片
    image_files = sorted(Path(image_dir).glob('*.jpg'))

    if len(image_files) == 0:
        print(f"错误: 在 {image_dir} 中没有找到图片")
        return

    # 限制数量
    n_samples = min(len(image_files), len(texts))

    print(f"\n找到 {len(image_files)} 张图片，处理前 {n_samples} 张")

    results_list = []

    for i in range(n_samples):
        image_path = str(image_files[i])
        text = texts[i] if i < len(texts) else "A photo"

        print(f"\n[{i+1}/{n_samples}] 处理: {os.path.basename(image_path)}")

        save_path = os.path.join(output_dir, f"causal_vis_{i:04d}.png")

        try:
            results = visualizer.visualize_single_sample(
                image_path=image_path,
                text=text,
                save_path=save_path
            )

            results_list.append({
                'image': os.path.basename(image_path),
                'text': text,
                'causal_effect': results['causal_effect'].item(),
                'shared_similarity': results['verification']['shared_similarity'],
                'all_satisfied': results['verification']['all_satisfied']
            })

        except Exception as e:
            print(f"错误处理 {image_path}: {e}")
            continue

    # 生成汇总报告
    print("\n" + "="*80)
    print("批量处理汇总")
    print("="*80)

    for r in results_list:
        status = "✅" if r['all_satisfied'] else "❌"
        print(f"{status} {r['image']}: 因果效应={r['causal_effect']:.4f}, 相似度={r['shared_similarity']:.4f}")

    print("\n统计:")
    satisfied_count = sum(1 for r in results_list if r['all_satisfied'])
    print(f"  前门准则满足: {satisfied_count}/{len(results_list)}")
    print(f"  平均因果效应: {sum(r['causal_effect'] for r in results_list)/len(results_list):.4f}")
    print(f"  平均相似度: {sum(r['shared_similarity'] for r in results_list)/len(results_list):.4f}")


def main():
    parser = argparse.ArgumentParser(description='批量因果链可视化')

    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/mscoco_captions/images',
        help='图片目录'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='输出目录'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='处理样本数量'
    )

    parser.add_argument(
        '--clip-model',
        type=str,
        default=None,
        help='CLIP 模型路径'
    )

    parser.add_argument(
        '--frontdoor-model',
        type=str,
        default=None,
        help='FrontDoor 模型路径'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu']
    )

    args = parser.parse_args()

    # 默认文本描述
    default_texts = [
        "A group of people dancing in a party",
        "A giraffe eating food from a tree",
        "A flower vase sitting on a table",
        "A zebra grazing on green grass",
        "Woman in swim suit holding a beach ball",
        "Closeup of bins of food that are brightly colored",
        "A meal is presented in brightly colored bins",
        "Containers filled with different food items",
        "Colorful dishes holding meat and vegetables",
        "Bunch of trays with different food"
    ] * 10  # 扩展到足够多

    batch_visualize(
        image_dir=args.image_dir,
        texts=default_texts[:args.num_samples],
        output_dir=args.output_dir,
        clip_model=args.clip_model,
        frontdoor_model=args.frontdoor_model,
        device=args.device
    )


if __name__ == "__main__":
    main()
