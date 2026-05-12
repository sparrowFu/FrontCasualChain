"""
测试所有模块导入是否正常
"""
import sys


def test_imports():
    """测试所有模块的导入"""
    print("测试模块导入...\n")

    tests = []

    # 测试 common 模块
    print("1. 测试 common 模块...")
    try:
        from common import BaseConfig, AvgMeter, get_lr
        from common import MSCOCOCaptionsDataset, load_mscoco_data, build_loaders, get_transforms
        from common import train_epoch, valid_epoch
        print("   ✅ common 模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ common 模块导入失败: {e}")
        tests.append(False)

    # 测试 CLIP 模型
    print("\n2. 测试 CLIP 模型...")
    try:
        from models.clip import CLIPConfig
        from models.clip.model import CLIPModel, ImageEncoder, TextEncoder, ProjectionHead
        from models.clip import train as train_clip
        from models.clip import evaluate as evaluate_clip
        print("   ✅ CLIP 模型导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ CLIP 模型导入失败: {e}")
        tests.append(False)

    # 测试 FrontDoor 模型
    print("\n3. 测试 FrontDoor 模型...")
    try:
        from models.frontdoor import FrontDoorConfig
        from models.frontdoor.model import FrontDoorCausalModel, FrontDoorWithEncoders
        from models.frontdoor.loss import FrontDoorLoss
        from models.frontdoor import train as train_frontdoor
        from models.frontdoor import evaluate as evaluate_frontdoor
        print("   ✅ FrontDoor 模型导入成功")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ FrontDoor 模型导入失败: {e}")
        tests.append(False)

    # 测试配置
    print("\n4. 测试配置...")
    try:
        from common.config import BaseConfig
        config = BaseConfig()
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'device')
        assert hasattr(config, 'dataset_path')
        assert hasattr(config, 'images_path')
        print(f"   ✅ 配置测试成功")
        print(f"      - batch_size: {config.batch_size}")
        print(f"      - device: {config.device}")
        print(f"      - dataset_path: {config.dataset_path}")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ 配置测试失败: {e}")
        tests.append(False)

    # 测试模型配置
    print("\n5. 测试模型配置...")
    try:
        from models.clip.config import CLIPConfig
        from models.frontdoor.config import FrontDoorConfig

        clip_config = CLIPConfig()
        frontdoor_config = FrontDoorConfig()

        assert hasattr(clip_config, 'model_name')
        assert hasattr(frontdoor_config, 'shared_dim')
        assert hasattr(frontdoor_config, 'private_ratio')

        print("   ✅ 模型配置测试成功")
        print(f"      - CLIP model_name: {clip_config.model_name}")
        print(f"      - FrontDoor shared_dim: {frontdoor_config.shared_dim}")
        tests.append(True)
    except Exception as e:
        print(f"   ❌ 模型配置测试失败: {e}")
        tests.append(False)

    # 测试模型创建（如果 torch 可用）
    print("\n6. 测试模型创建...")
    try:
        import torch
        from models.clip.model import CLIPModel

        model = CLIPModel()
        assert isinstance(model, torch.nn.Module)
        print("   ✅ CLIP 模型创建成功")
        tests.append(True)
    except ImportError:
        print("   ⚠️  PyTorch 未安装，跳过模型创建测试")
        tests.append(True)  # 不计入失败
    except Exception as e:
        print(f"   ❌ 模型创建测试失败: {e}")
        tests.append(False)

    # 检查已删除的模块无法导入
    print("\n7. 检查旧模块已移除...")
    old_modules = [
        'common.BaseDataset',
        'common.data',
        'common.dataset_loaders',
    ]
    all_removed = True
    for module in old_modules:
        try:
            __import__(module)
            print(f"   ⚠️  {module} 仍然可导入（应该已删除）")
            all_removed = False
        except ImportError:
            pass  # 期望的行为

    if all_removed:
        print("   ✅ 所有旧模块已正确移除")
        tests.append(True)
    else:
        print("   ❌ 部分旧模块仍然存在")
        tests.append(False)

    # 总结
    print("\n" + "=" * 50)
    passed = sum(tests)
    total = len(tests)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print(f"⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
