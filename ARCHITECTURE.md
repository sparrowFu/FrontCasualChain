# 架构说明文档

## 设计理念

本项目采用模块化、可扩展的设计理念，便于：
- 快速添加新的模型
- 代码复用和维护
- 统一的训练和评估接口
- 简化的数据集支持

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    FrontdoorCausalChain                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ train.py    │  │ evaluate.py  │  │  README.md   │      │
│  │ 统一训练入口  │  │ 统一评估入口  │  │  项目文档    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                                  │
│         └─────────┬───────┘                                  │
│                   │                                          │
│  ┌────────────────▼──────────────────────────────────┐      │
│  │                    models/                         │      │
│  │              各模型实现目录                         │      │
│  ├────────────────────────────────────────────────────┤      │
│  │                                                     │      │
│  │  ┌─────────────┐  ┌──────────────┐                │      │
│  │  │    clip/    │  │  frontdoor/  │                │      │
│  │  │  CLIP模型   │  │  因果链模型   │                │      │
│  │  │             │  │              │                │      │
│  │  │ ├config.py  │  │ ├config.py   │                │      │
│  │  │ ├model.py   │  │ ├model.py    │                │      │
│  │  │ ├train.py   │  │ ├loss.py     │                │      │
│  │  │ └evaluate.py│  │ ├train.py    │                │      │
│  │  └─────────────┘  │ └evaluate.py │                │      │
│  │                   └──────────────┘                │      │
│  └─────────────────────────────────────────────────────┘      │
│                   │ 依赖                                     │
│                   ▼                                          │
│  ┌─────────────────────────────────────────────────────┐     │
│  │                    common/                          │     │
│  │              共享工具和基础类                         │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │                                                      │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │     │
│  │  │  config.py  │  │  dataset.py │  │  metrics.py │ │     │
│  │  │  BaseConfig │  │ MSCOCO数据集 │  │ AvgMeter    │ │     │
│  │  │             │  │             │  │ get_lr      │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │     │
│  │                                                      │     │
│  │  ┌─────────────────────────────┐                    │     │
│  │  │        training.py          │                    │     │
│  │  │  train_epoch()              │                    │     │
│  │  │  valid_epoch()              │                    │     │
│  │  └─────────────────────────────┘                    │     │
│  │                                                      │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                        data/                        │    │
│  │                   MSCOCO Captions                   │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  mscoco_captions/                                    │    │
│  │  ├── captions/     (parquet 格式)                   │    │
│  │  ├── images/       (图片文件)                       │    │
│  │  ├── train/        (VQA 数据，暂未使用)             │    │
│  │  └── test/         (VQA 数据，暂未使用)             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 模块依赖关系

```
train.py
    │
    ├──> models.frontdoor
    │        │
    │        ├──> common (共享工具)
    │        └──> models.clip (编码器)
    │             └──> common (共享工具)
    │
    └──> models.clip
             │
             ├──> common (共享工具)
             └──> transformers, timm (外部库)
```

## 数据流

### 数据集加载流程

```
1. 读取 parquet 文件 (load_mscoco_data)
   ├─> captions/train-00000-of-00001.parquet
   └─> 包含: url, caption, image_file_name
   ↓
2. 构建 DataFrame
   ├─> 列: image_file_name, caption (list[str])
   └─> 118,287 条数据
   ↓
3. 划分训练集/验证集
   ├─> 默认 80/20 划分
   └─> 可配置 test_size
   ↓
4. 构建 DataLoader (build_loaders)
   └─> MSCOCOCaptionsDataset
   ↓
5. 训练时加载
   ├─> 从 images/ 读取图片
   └─> 从 parquet 读取 caption
```

### FrontDoor 训练流程

```
1. 读取数据 (load_mscoco_data)
   └─> 返回 train_df, valid_df
   ↓
2. 创建数据加载器 (build_loaders)
   └─> 返回 train_loader, valid_loader
   ↓
3. 加载预训练编码器 (ImageEncoder, TextEncoder)
   ↓
4. 初始化因果模型 (FrontDoorCausalModel)
   ↓
5. 训练循环 (train_epoch)
   │
   ├──> 前向传播 (model.forward)
   │    ├─> 图像编码
   │    ├─> 文本编码
   │    ├─> Shared/Private 分解
   │    ├─> 共享语义计算
   │    └─> 因果效应估计
   │
   ├──> 计算损失 (FrontDoorLoss)
   │    ├─> 对齐损失 (alignment_loss)
   │    ├─> 正交损失 (orthogonal_loss)
   │    ├─> 对比损失 (contrastive_loss)
   │    └─> 重建损失 (reconstruction_loss)
   │
   ├──> 反向传播
   └──> 更新参数
   ↓
6. 验证 (valid_epoch)
   ↓
7. 保存最佳模型
```

## 类继承关系

```
BaseConfig (common/config.py)
    │
    ├──> CLIPConfig (models/clip/config.py)
    └──> FrontDoorConfig (models/frontdoor/config.py)

MSCOCOCaptionsDataset (common/dataset.py)
    │
    └──> 用于 MSCOCO Captions 数据集
        ├─> 从 parquet 加载元数据
        └─> 从 images/ 加载图片
```

## 数据集格式

### MSCOCO Captions

```
数据集结构:
mscoco_captions/
├── captions/
│   └── train-00000-of-00001.parquet
│       ├── url: 图片 URL
│       ├── caption: list[str] (多条文本描述)
│       └── image_file_name: 图片文件名
├── images/
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ... (118,287 个图片文件)
├── train/ (VQA 微调数据，暂未使用)
│   ├── img_ids.txt
│   ├── questions.txt
│   └── answers.txt
└── test/ (VQA 微调数据，暂未使用)
    ├── img_ids.txt
    ├── questions.txt
    └── answers.txt

Parquet 文件格式:
+-------------------------+----------------------------------+---------------------+
| url                     | caption                          | image_file_name     |
+-------------------------+----------------------------------+---------------------+
| http://images.cocod...  | ["Closeup of bins...", "A lar...] | 000000000009.jpg   |
| http://images.cocod...  | ["A giraffe eating...", "Two g...] | 000000000025.jpg   |
+-------------------------+----------------------------------+---------------------+
```

## 扩展指南

### 添加新模型

参考 `models/clip/` 或 `models/frontdoor/` 目录，实现以下文件：

- `config.py` - 模型配置（继承 `BaseConfig`）
- `model.py` - 模型定义
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本

### 添加自定义损失函数

参考 `models/frontdoor/loss.py`：

```python
class CustomLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, output, targets):
        # 计算损失
        return loss

    def get_metrics(self, output):
        # 返回评估指标
        return metrics
```

## 最佳实践

### 1. 配置管理

- 使用配置类集中管理参数
- 支持命令行参数覆盖
- 记录实验配置

### 2. 代码复用

- 优先使用 `common` 中的工具
- 避免重复造轮子
- 保持接口一致

### 3. 模块化设计

- 单一职责原则
- 清晰的模块边界
- 易于测试和维护

## 性能考虑

### 内存优化

- 使用 DataLoader 的多进程加载
- 适当的 batch_size
- 及时清理无用变量

### 计算优化

- 混合精度训练
- 梯度累积
- 学习率调度

### I/O 优化

- Parquet 格式高效读取
- 异步数据加载
- 图像变换优化
