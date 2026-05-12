# 文件索引

本文档提供了项目中所有文件的索引和说明。

## 根目录文件

### 文档文件

| 文件 | 说明 |
|------|------|
| [README.md](README.md) | 项目主文档 |
| [QUICKSTART.md](QUICKSTART.md) | 快速开始指南 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构详细说明 |
| [INDEX.md](INDEX.md) | 文件索引（本文件） |

### 入口脚本

| 文件 | 说明 |
|------|------|
| [train.py](train.py) | 统一训练入口 |
| [evaluate.py](evaluate.py) | 统一评估入口 |

### 工具脚本

| 文件 | 说明 |
|------|------|
| [check_structure.py](check_structure.py) | 项目结构检查 |
| [test_imports.py](test_imports.py) | 导入测试 |

## common/ 目录

共享工具和基础类。

| 文件 | 说明 |
|------|------|
| [__init__.py](common/__init__.py) | 模块导出 |
| [config.py](common/config.py) | 基础配置类 BaseConfig |
| [dataset.py](common/dataset.py) | MSCOCO 数据集类和加载函数 |
| [metrics.py](common/metrics.py) | 评估指标工具 AvgMeter, get_lr |
| [training.py](common/training.py) | 训练和验证函数 |

### 核心类说明

- **BaseConfig**: 所有模型的基类配置，包含数据集路径配置
- **MSCOCOCaptionsDataset**: MSCOCO Captions 数据集类
- **load_mscoco_data**: 加载并划分数据集
- **build_loaders**: 构建 PyTorch DataLoader

## models/ 目录

模型实现目录。

### models/clip/

CLIP 模型实现。

| 文件 | 说明 |
|------|------|
| [__init__.py](models/clip/__init__.py) | 模块导出 |
| [config.py](models/clip/config.py) | CLIP 配置类 |
| [model.py](models/clip/model.py) | CLIP 模型定义 |
| [train.py](models/clip/train.py) | CLIP 训练脚本 |
| [evaluate.py](models/clip/evaluate.py) | CLIP 评估脚本 |

**核心类**:
- `ImageEncoder`: 图像编码器 (ResNet50)
- `TextEncoder`: 文本编码器 (DistilBERT)
- `ProjectionHead`: 投影头
- `CLIPModel`: CLIP 主模型

### models/frontdoor/

FrontDoor 因果链模型实现。

| 文件 | 说明 |
|------|------|
| [__init__.py](models/frontdoor/__init__.py) | 模块导出 |
| [config.py](models/frontdoor/config.py) | FrontDoor 配置类 |
| [model.py](models/frontdoor/model.py) | FrontDoor 模型定义 |
| [loss.py](models/frontdoor/loss.py) | FrontDoor 损失函数 |
| [train.py](models/frontdoor/train.py) | FrontDoor 训练脚本 |
| [evaluate.py](models/frontdoor/evaluate.py) | FrontDoor 评估脚本 |

**核心类**:
- `FrontDoorCausalModel`: 因果链模型
- `FrontDoorWithEncoders`: 包含编码器的完整模型
- `FrontDoorLoss`: 组合损失函数

**损失函数**:
- 对齐损失 (alignment_loss)
- 正交损失 (orthogonal_loss)
- 对比损失 (contrastive_loss)
- 重建损失 (reconstruction_loss)

## data/ 目录

数据集存储目录。

### data/mscoco_captions/

MSCOCO Captions 数据集。

```
mscoco_captions/
├── captions/              # Parquet 格式元数据
│   └── train-00000-of-00001.parquet
│       ├── url            # 图片 URL
│       ├── caption        # list[str] 文本描述
│       └── image_file_name # 图片文件名
├── images/                # 图片文件
│   ├── 000000000009.jpg
│   └── ... (118,287 个文件)
├── train/                 # VQA 训练数据 (暂未使用)
│   ├── img_ids.txt
│   ├── questions.txt
│   └── answers.txt
└── test/                  # VQA 测试数据 (暂未使用)
    ├── img_ids.txt
    ├── questions.txt
    └── answers.txt
```

## PreTrainedModels/ 目录

预训练模型存储目录。

```
PreTrainedModels/
└── distilbert_base_uncased/    # DistilBERT 模型
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## results/ 目录

训练结果输出目录（训练时自动创建）。

```
results/
├── clipmodel/
│   ├── best_model.pt          # 最佳模型权重
│   └── checkpoint.pt          # 训练检查点
└── frontdoormodel/
    ├── best_model.pt          # 最佳模型权重
    └── checkpoint.pt          # 训练检查点
```

## 文件统计

| 类型 | 数量 |
|------|------|
| Python 文件 | 20+ |
| Markdown 文档 | 4 |
| 模块 | 2 (clip, frontdoor) |
| 支持的数据集 | 1 (mscoco_captions) |

## 依赖关系图

```
train.py
    ├── models.frontdoor
    │   ├── common.config
    │   ├── common.dataset
    │   ├── common.training
    │   ├── models.clip (编码器)
    │   └── common.metrics
    └── models.clip
        ├── common.config
        ├── common.dataset
        ├── common.training
        └── common.metrics
```

## 快速导航

### 添加新模型
1. 参考 `models/clip/` 或 `models/frontdoor/`
2. 创建新的模型目录
3. 实现配置、模型、训练和评估脚本
4. 更新 `models/__init__.py`

### 修改训练配置
1. 修改 `models/*/config.py` 中的模型配置
2. 或修改 `common/config.py` 中的基础配置

### 运行训练
```bash
python train.py --model clip
python train.py --model frontdoor
```
