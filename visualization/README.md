# 前门准则因果链可视化模块

本模块用于验证和可视化 FrontDoor 模型的因果推断过程，提供多模态特征分析和可视化工具。

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

4. **多模态空间可视化**
   - 高维特征降维展示（t-SNE、PCA）
   - 聚类分析
   - 相似度分布图

## 文件结构

```
visualization/
├── __init__.py                    # 模块初始化
├── causal_visualizer.py           # 核心可视化类
├── example.py                     # 单样本可视化示例
├── batch_visualize.py             # 批量可视化脚本
├── visualize_multi_modal_space.py # 多模态空间可视化
└── README.md                      # 本文件
```

## 模块说明

### causal_visualizer.py

核心可视化类 `CausalChainVisualizer`，提供完整的因果链分析功能。

**主要方法**:
- `visualize_single_sample()` - 单样本可视化
- `verify_frontdoor_criteria()` - 前门准则验证
- `compute_causal_effect()` - 因果效应计算
- `print_verification_report()` - 验证报告输出

### example.py

命令行工具，用于单样本因果链可视化。

### batch_visualize.py

批量处理工具，用于多样本分析和统计。

### visualize_multi_modal_space.py

多模态特征空间可视化工具，支持高维特征降维和聚类分析。

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

### 方式4: 多模态空间可视化

```bash
# 使用默认参数
python visualization/visualize_multi_modal_space.py

# 自定义参数
python visualization/visualize_multi_modal_space.py \
    --num-samples 500 \
    --method tsne \
    --output-dir results/space_visualization
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

### visualize_multi_modal_space.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-samples` | 处理样本数 | 200 |
| `--method` | 降维方法 (tsne/pca) | tsne |
| `--output-dir` | 输出目录 | results/space_viz |
| `--clip-model` | CLIP 模型路径 | auto |
| `--frontdoor-model` | FrontDoor 模型路径 | auto |
| `--device` | 设备选择 | auto |

## 可视化输出

### 因果链可视化图表

生成的图表包含 6 个子图：

1. **原始编码特征**：图像和文本的原始编码特征对比
2. **Shared 特征**：分解后的共同语义特征
3. **Private 特征**：模态私有特征
4. **共享语义热图**：前 64 维共享语义的热力图
5. **Shared 特征相似度**：相似度值及阈值对比
6. **因果效应值**：计算的因果效应

### 多模态空间可视化图表

生成的图表展示：

1. **t-SNE/PCA 降维图**：高维特征的二维投影
   - 图像特征分布
   - 文本特征分布
   - Shared 特征分布
   - Private 特征分布

2. **聚类分析**：
   - K-means 聚类结果
   - 类内相似度
   - 类间分离度

3. **相似度分布**：
   - 图文相似度直方图
   - Shared-Private 正交性验证

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
- scikit-learn（用于多模态空间可视化）
- numpy

## 示例输出

### 单样本可视化

执行 `example.py` 后，将生成包含 6 个子图的可视化图表，展示：
- 特征分解效果
- 前门准则验证结果
- 计算的因果效应值

### 批量可视化

执行 `batch_visualize.py` 后，将在输出目录生成：
- 每个样本的可视化图表
- 统计汇总报告
- 验证通过率

### 空间可视化

执行 `visualize_multi_modal_space.py` 后，将生成：
- 特征空间散点图（t-SNE/PCA）
- 聚类分析图
- 相似度分布直方图
