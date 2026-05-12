# 快速开始指南

本指南将帮助你快速上手 FrontdoorCausalChain 项目。

## 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU 加速)

## 安装步骤

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers timm albumentations scikit-learn pandas opencv-python matplotlib tqdm pyarrow
```

### 2. 准备数据集

项目使用 MSCOCO Captions 数据集：

```bash
# 数据集结构 (已提供):
# data/mscoco_captions/
# ├── captions/
# │   └── train-00000-of-00001.parquet  (118,287 条)
# ├── images/                            (118,287 个图片文件)
# ├── train/                             (VQA 数据，暂未使用)
# └── test/                              (VQA 数据，暂未使用)
```

### 3. 准备预训练模型

预训练模型已包含在 `PreTrainedModels/` 目录中：

```
PreTrainedModels/
└── distilbert_base_uncased/
```

## 快速运行

### 训练 CLIP 模型

```bash
# 使用默认配置
python train.py --model clip

# 调整训练参数
python train.py --model clip --batch-size 64 --epochs 10

# 调试模式（快速验证代码）
python train.py --model clip --debug
```

### 训练 FrontDoor 模型

```bash
# 使用默认配置
python train.py --model frontdoor

# 调整训练参数
python train.py --model frontdoor --batch-size 32 --epochs 5 --lr 1e-5

# 调试模式
python train.py --model frontdoor --debug
```

### 评估模型

```bash
# 评估 CLIP 模型（图文检索）
python evaluate.py --model clip --query "a beautiful sunset"

# 评估 FrontDoor 模型
python evaluate.py --model frontdoor --num-samples 100
```

## 命令行参数说明

### train.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型选择 (clip/frontdoor) | clip |
| `--batch-size` | 批大小 | 32 |
| `--epochs` | 训练轮数 | 1 (clip) / 2 (frontdoor) |
| `--lr` | 学习率 (仅 frontdoor) | 1e-5 |
| `--debug` | 启用调试模式 | False |
| `--no-resume` | 不从 checkpoint 恢复 | False |

示例：

```bash
# 完整参数示例
python train.py --model frontdoor --batch-size 64 --epochs 10 --lr 1e-4
```

## 配置文件

### 修改默认配置

编辑 `models/frontdoor/config.py`：

```python
class FrontDoorConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # 模型参数
        self.shared_dim = 256
        self.private_ratio = 0.3

        # 训练参数
        self.batch_size = 32
        self.epochs = 2
        self.lr = 1e-5

        # 损失权重
        self.lambda_alignment = 1.0
        self.lambda_orthogonal = 0.1
        self.lambda_contrastive = 1.0
        self.lambda_reconstruction = 0.5
```

### 修改基础配置

编辑 `common/config.py` 中的 `BaseConfig` 类，所有模型都会继承这些配置。

## 常见问题

### Q1: 如何切换模型？

**A:** 使用 `--model` 参数：

```bash
python train.py --model frontdoor
```

### Q2: 如何使用 GPU？

**A:** 确保安装了 CUDA 版本的 PyTorch，代码会自动检测 GPU。

### Q3: 内存不足怎么办？

**A:** 减小 batch_size：

```bash
python train.py --model clip --batch-size 16
```

### Q4: 如何快速验证代码？

**A:** 使用调试模式，只使用少量数据：

```bash
python train.py --model clip --debug
```

### Q5: 数据集路径不对怎么办？

**A:** 检查 `common/config.py` 中的 `project_root` 配置，确保路径与你的实际目录一致：

```python
project_root = "D:\\code\\causality\\FrontdoorCausalChain"
```

## 模型输出

训练完成后，模型保存在 `results/` 目录：

```
results/
├── clipmodel/
│   ├── best_model.pt      # 最佳模型
│   └── checkpoint.pt      # 训练检查点
└── frontdoormodel/
    ├── best_model.pt      # 最佳模型
    └── checkpoint.pt      # 训练检查点
```

## 进阶用法

### 实验记录

建议使用实验跟踪工具记录实验结果：

```python
import wandb

# 在训练开始前
wandb.init(project="my-project", config=config.__dict__)

# 在训练循环中
wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
```

### 多GPU训练

使用 PyTorch 的 `DistributedDataParallel`：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

## 下一步

- 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解项目架构
- 查看 [models/frontdoor/](models/frontdoor/) 了解模型实现
- 尝试修改模型配置进行实验
