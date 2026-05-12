# FrontdoorCausalChain

基于前门准则的多模态因果链学习项目，支持图文检索、因果推断等多种任务。

## 项目概述

本项目实现了多种多模态模型，包括：

- **CLIP 模型**: 基于对比学习的图文检索模型
- **FrontDoor 因果链模型**: 基于前门准则的因果推断模型

## 目录结构

```
FrontdoorCausalChain/
├── common/                    # 共享工具和配置
│   ├── __init__.py
│   ├── config.py             # 基础配置类
│   ├── dataset.py            # MSCOCO 数据集类
│   ├── metrics.py            # 评估指标工具
│   └── training.py           # 训练和验证函数
│
├── models/                    # 模型实现目录
│   ├── __init__.py
│   ├── clip/                 # CLIP 模型
│   │   ├── __init__.py
│   │   ├── config.py         # CLIP 配置
│   │   ├── model.py          # CLIP 模型定义
│   │   ├── train.py          # CLIP 训练脚本
│   │   └── evaluate.py       # CLIP 评估脚本
│   └── frontdoor/            # FrontDoor 因果链模型
│       ├── __init__.py
│       ├── config.py         # FrontDoor 配置
│       ├── model.py          # FrontDoor 模型定义
│       ├── loss.py           # FrontDoor 损失函数
│       ├── train.py          # FrontDoor 训练脚本
│       └── evaluate.py       # FrontDoor 评估脚本
│
├── data/                      # 数据集目录
│   └── mscoco_captions/      # MSCOCO Captions 数据集
│       ├── captions/         # Parquet 格式元数据
│       ├── images/           # 图片文件
│       ├── train/            # VQA 训练数据
│       └── test/             # VQA 测试数据
│
├── PreTrainedModels/          # 预训练模型
│   └── distilbert_base_uncased/
│
├── train.py                   # 统一训练入口
├── evaluate.py                # 统一评估入口
└── results/                   # 训练结果输出
```

## 数据集

项目使用 **MSCOCO Captions** 数据集：

| 组件 | 说明 | 数量 |
|------|------|------|
| 图片 | MSCOCO 训练集图片 | 118,287 |
| 描述 | 每张图片多条文本描述 | 平均 5 条/图 |

数据集结构：
```
mscoco_captions/
├── captions/train-00000-of-00001.parquet
├── images/*.jpg
├── train/ (VQA 数据，暂未使用)
└── test/ (VQA 数据，暂未使用)
```

## 快速开始

### 安装依赖

```bash
pip install torch transformers timm albumentations sklearn pandas opencv-python matplotlib tqdm pyarrow
```

### 训练模型

```bash
# 训练 CLIP 模型
python train.py --model clip

# 训练 FrontDoor 模型
python train.py --model frontdoor

# 自定义训练参数
python train.py --model clip --batch-size 64 --epochs 10

# 启用调试模式（使用少量数据）
python train.py --model clip --debug
```

### 评估模型

```bash
# 评估 CLIP 模型
python evaluate.py --model clip --query "a beautiful sunset"

# 评估 FrontDoor 模型
python evaluate.py --model frontdoor --num-samples 100
```

## 配置说明

### 基础配置

所有模型都继承自 `BaseConfig`：

```python
from common.config import BaseConfig

config = BaseConfig()
# 数据集路径（固定为 mscoco_captions）
config.dataset_path  # 数据集根目录
config.images_path   # 图片目录
config.captions_path # 元数据目录
```

### 模型配置

每个模型都有独立的配置类：

```python
from models.frontdoor.config import FrontDoorConfig

config = FrontDoorConfig()
# 覆盖默认配置
config.shared_dim = 512
config.batch_size = 64
config.epochs = 20
```

## 模型说明

### CLIP 模型

基于对比学习的图文检索模型，使用 ResNet50 和 DistilBERT 作为编码器。

- **输入**: 图像 + 文本
- **输出**: 图文相似度
- **损失函数**: 对比学习损失

### FrontDoor 因果链模型

基于前门准则的因果推断模型，将特征分解为 shared 和 private 部分。

- **输入**: 图像 + 文本
- **输出**: 因果效应值
- **损失函数**: 组合损失（对齐、正交、对比、重建）

**模型特点**:
- Shared/Private 特征分解
- 共享语义融合
- 因果效应估计
- 多重损失优化

## 添加新模型

### 实现步骤

```python
# 1. 创建配置类 (models/your_model/config.py)
from common.config import BaseConfig

class YourModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'your_model'
        # 添加模型特定配置

# 2. 实现模型 (models/your_model/model.py)
class YourModel(nn.Module):
    def __init__(self, config):
        # 实现模型逻辑
        pass

# 3. 实现训练脚本 (models/your_model/train.py)
def train(config=None):
    # 实现训练逻辑
    pass

# 4. 更新 models/__init__.py
from . import your_model
```

## 命令行参数

### train.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型选择 (clip/frontdoor) | clip |
| `--batch-size` | 批大小 | 配置文件值 |
| `--epochs` | 训练轮数 | 配置文件值 |
| `--lr` | 学习率 | 配置文件值 |
| `--debug` | 启用调试模式 | False |
| `--no-resume` | 不从 checkpoint 恢复 | False |

### evaluate.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型选择 (clip/frontdoor) | clip |
| `--query` | 查询文本（CLIP） | "a group of people..." |
| `--num-samples` | 评估样本数（FrontDoor） | 100 |

## 依赖项

- PyTorch >= 1.10
- transformers >= 4.20
- timm >= 0.6
- albumentations >= 1.0
- scikit-learn
- pandas
- pyarrow
- opencv-python
- matplotlib
- tqdm

## 文档

- [README.md](README.md) - 项目说明（本文件）
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构详细说明
- [INDEX.md](INDEX.md) - 文件索引

## 许可证

本项目仅用于学术研究和教育目的。
