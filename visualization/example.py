"""
前门准则因果链可视化示例

使用方法:
    python visualization/example.py --image data/mscoco_captions/images/000000000009.jpg --text "A group of people dancing in a party"
    python visualization/example.py --image data/mscoco_captions/images/000000000025.jpg --text "A giraffe eating food" --save output.png
"""
import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from visualization.causal_visualizer import CausalChainVisualizer


def main():
    parser = argparse.ArgumentParser(description='前门准则因果链可视化')

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='图像路径'
    )

    parser.add_argument(
        '--text',
        type=str,
        default='A group of people dancing in a party',
        help='文本描述'
    )

    parser.add_argument(
        '--clip-model',
        type=str,
        default=None,
        help='CLIP 模型路径（可选）'
    )

    parser.add_argument(
        '--frontdoor-model',
        type=str,
        default=None,
        help='FrontDoor 模型路径（可选）'
    )

    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='可视化保存路径（可选）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='设备选择'
    )

    args = parser.parse_args()

    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return 1

    # 设置默认模型路径
    if args.clip_model is None:
        args.clip_model = os.path.join(
            project_root,
            'results',
            'clipmodel',
            'best_model.pt'
        )

    if args.frontdoor_model is None:
        args.frontdoor_model = os.path.join(
            project_root,
            'results',
            'frontdoormodel',
            'best_model.pt'
        )

    # 创建可视化器
    print("初始化因果链可视化器...")
    visualizer = CausalChainVisualizer(
        clip_model_path=args.clip_model if os.path.exists(args.clip_model) else None,
        frontdoor_model_path=args.frontdoor_model if os.path.exists(args.frontdoor_model) else None,
        device=args.device
    )

    # 执行可视化
    print("\n开始因果推断和可视化...")
    results = visualizer.visualize_single_sample(
        image_path=args.image,
        text=args.text,
        save_path=args.save
    )

    # 打印验证报告
    visualizer.print_verification_report(results['verification'])

    return 0


if __name__ == "__main__":
    sys.exit(main())
