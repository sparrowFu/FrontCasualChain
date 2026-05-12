"""
评估脚本 - 统一的评估入口

运行方式:
    python evaluate.py
    python evaluate.py --model clip
    python evaluate.py --model frontdoor
    python evaluate.py --model clip --query "a beautiful sunset"
"""
import argparse
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_clip(args):
    """评估 CLIP 模型"""
    from models.clip import evaluate as clip_evaluate

    print("=" * 60)
    print("评估 CLIP 模型")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"查询文本: {args.query}")

    # 调用 CLIP 评估函数
    clip_evaluate.evaluate(model_path=args.model_path, query=args.query)


def evaluate_frontdoor(args):
    """评估 FrontDoor 因果链模型"""
    from models.frontdoor.evaluate import evaluate
    from models.frontdoor.config import FrontDoorConfig

    # 创建配置
    config = FrontDoorConfig()

    if args.model_path is not None:
        config.model_save_path = args.model_path
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = 100

    print("=" * 60)
    print("评估 FrontDoor 因果链模型")
    print("=" * 60)
    print(f"模型路径: {config.model_save_path}")
    print(f"评估样本数: {num_samples}")

    # 调用 FrontDoor 评估函数
    evaluate(config=config, num_samples=num_samples)


def main():
    parser = argparse.ArgumentParser(
        description='评估多模态模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate.py                                  # 评估 CLIP 模型（默认）
  python evaluate.py --model frontdoor                # 评估 FrontDoor 模型
  python evaluate.py --model clip --query "a beautiful sunset"
  python evaluate.py --model frontdoor --num-samples 50
        """
    )

    # 模型选择
    parser.add_argument(
        '--model',
        type=str,
        default='clip',
        choices=['clip', 'frontdoor'],
        help='选择要评估的模型 (clip/frontdoor)'
    )

    # 模型参数
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='模型权重路径'
    )

    # 评估参数
    parser.add_argument(
        '--query',
        type=str,
        default='a group of people dancing in a party',
        help='查询文本（用于 CLIP 模型）'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='评估样本数量（用于 FrontDoor 模型）'
    )

    args = parser.parse_args()

    # 设置默认模型路径
    if args.model_path is None:
        if args.model == 'clip':
            args.model_path = 'best.pt'
        elif args.model == 'frontdoor':
            from models.frontdoor.config import FrontDoorConfig
            config = FrontDoorConfig()
            args.model_path = config.model_save_path

    # 根据模型类型调用相应的评估函数
    if args.model == 'clip':
        evaluate_clip(args)
    elif args.model == 'frontdoor':
        evaluate_frontdoor(args)
    else:
        print(f"未知模型: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
