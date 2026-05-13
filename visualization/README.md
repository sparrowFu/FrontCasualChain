# 前门准则因果链可视化模块

本模块用于验证和可视化 FrontDoor 模型的因果推断过程。

## 功能特性

1. **特征分解可视化**
   - 原始编码特征
   - Shared 特征（共同语义）
   - Private 特征（模态私有）

2. **前门准则验证**
   - 条件1：完全中介（Shared 特征相关性）
   - 条件2：I,Q → M 无混杂（编码过程独立性）
   - 条件3：M → A 无混杂（Shared-Private 正交性）

3. **因果效应计算**
   - 图像 → 文本的因果效应值

## 文件结构

```
visualization/
├── __init__.py              # 模块初始化
├── causal_visualizer.py     # 核心可视化类
├── example.py               # 单样本可视化示例
├── batch_visualize.py       # 批量可视化脚本
└── README.md                # 本文件
```

## 使用方法

### 方式1: 单样本可视化（Python API）

```python
from visualization import CausalChainVisualizer

# 初始化可视化器
visualizer = CausalChainVisualizer(
    clip_model_path='results/clipmodel/best_model.pt',
    frontdoor_model_path='results/frontdoormodel/best_model.pt'
)

# 可视化单样本
results = visualizer.visualize_single_sample(
    image_path='data/mscoco_captions/images/000000000009.jpg',
    text='A group of people dancing in a party',
    save_path='causal_chain_visualization.png'
)

# 打印验证报告
visualizer.print_verification_report(results['verification'])
```

### 方式2: 单样本可视化（命令行）

```bash
python visualization/example.py \
    --image data/mscoco_captions/images/000000000009.jpg \
    --text "A group of people dancing in a party" \
    --save output.png
```

### 方式3: 批量可视化

```bash
python visualization/batch_visualize.py \
    --image-dir data/mscoco_captions/images \
    --output-dir results/visualizations \
    --num-samples 10
```

## 命令行参数

### example.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 图像路径（必需） | - |
| `--text` | 文本描述 | "A group of people..." |
| `--clip-model` | CLIP 模型路径 | results/clipmodel/best_model.pt |
| `--frontdoor-model` | FrontDoor 模型路径 | results/frontdoormodel/best_model.pt |
| `--save` | 保存路径 | 不保存 |
| `--device` | 设备选择 | auto |

### batch_visualize.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image-dir` | 图片目录 | data/mscoco_captions/images |
| `--output-dir` | 输出目录 | results/visualizations |
| `--num-samples` | 处理样本数 | 10 |
| `--clip-model` | CLIP 模型路径 | auto |
| `--frontdoor-model` | FrontDoor 模型路径 | auto |
| `--device` | 设备选择 | auto |

## 可视化输出

### 图表说明

生成的图表包含 6 个子图：

1. **原始编码特征**：图像和文本的原始编码特征对比
2. **Shared 特征**：分解后的共同语义特征
3. **Private 特征**：模态私有特征
4. **共享语义热图**：前 64 维共享语义的热力图
5. **Shared 特征相似度**：相似度值及阈值对比
6. **因果效应值**：计算的因果效应

### 前门准则验证

| 条件 | 含义 | 阈值 |
|------|------|------|
| 完全中介 | Shared 特征相关性 | > 0.5 |
| 无混杂 (I,Q→M) | 编码过程独立性 | 设计满足 |
| 无混杂 (M→A) | Shared-Private 正交性 | < 0.3 |

## 依赖要求

- PyTorch
- Matplotlib
- Seaborn
- OpenCV
- Transformers
- Albumentations
