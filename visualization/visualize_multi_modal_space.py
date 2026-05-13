"""
多模态特征空间可视化
将不同维度的text和image特征映射到同一向量空间并可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F


def visualize_multi_modal_space(text_features, image_features,
                                  title='多模态特征空间可视化',
                                  figsize=(12, 10),
                                  save_path=None):
    """
    将text和image特征映射到同一向量空间并可视化

    Args:
        text_features: list of torch.Tensor, 每个形状为 (1, 768)
        image_features: list of torch.Tensor, 每个形状为 (1, 2048)
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径（可选）

    Returns:
        fig: matplotlib图表对象
    """
    # 1. 转换为numpy数组并降维
    text_feats = []
    for t in text_features:
        if isinstance(t, torch.Tensor):
            feat = t.squeeze(0).detach().cpu().numpy()
        else:
            feat = np.array(t).squeeze()
        text_feats.append(feat)

    image_feats = []
    for i in image_features:
        if isinstance(i, torch.Tensor):
            feat = i.squeeze(0).detach().cpu().numpy()
        else:
            feat = np.array(i).squeeze()
        image_feats.append(feat)

    text_feats = np.array(text_feats)  # (25, 768)
    image_feats = np.array(image_feats)  # (5, 2048)

    # 确保是2D数组
    if text_feats.ndim == 1:
        text_feats = text_feats.reshape(1, -1)
    if image_feats.ndim == 1:
        image_feats = image_feats.reshape(1, -1)

    # 2. 维度对齐：使用池化将image特征转换为text特征维度
    text_dim = text_feats.shape[1]  # 768
    image_dim = image_feats.shape[1]  # 2048

    # 如果image维度更高，使用池化将其降到text维度
    if image_dim > text_dim:
        # 将每个image特征从2048维池化到768维
        image_feats_aligned = np.zeros((len(image_features), text_dim))
        for i, img_feat in enumerate(image_feats):
            img_feat_tensor = torch.tensor(img_feat).float().unsqueeze(0)  # [1, 2048]
            img_feat_pooled = F.adaptive_avg_pool1d(img_feat_tensor, text_dim).squeeze(0).numpy()  # [768]
            image_feats_aligned[i] = img_feat_pooled
        text_feats_aligned = text_feats
    # 如果text维度更高，使用池化将text降到image维度
    elif text_dim > image_dim:
        text_feats_aligned = np.zeros((len(text_features), image_dim))
        for i, txt_feat in enumerate(text_feats):
            txt_feat_tensor = torch.tensor(txt_feat).float().unsqueeze(0)  # [1, 768]
            txt_feat_pooled = F.adaptive_avg_pool1d(txt_feat_tensor, image_dim).squeeze(0).numpy()  # [2048]
            text_feats_aligned[i] = txt_feat_pooled
        image_feats_aligned = image_feats
    # 维度相同
    else:
        text_feats_aligned = text_feats
        image_feats_aligned = image_feats

    # 3. 合并所有特征
    all_feats = np.vstack([text_feats_aligned, image_feats_aligned])  # (30, target_dim)

    # 4. PCA降维到2D用于可视化
    pca_2d = PCA(n_components=2)
    feats_2d = pca_2d.fit_transform(all_feats)

    # 分离text和image的2D坐标
    text_2d = feats_2d[:len(text_features)]  # (25, 2)
    image_2d = feats_2d[len(text_features):]  # (5, 2)

    # 5. 计算中心点
    text_center = text_2d.mean(axis=0)
    image_center = image_2d.mean(axis=0)

    # 6. 创建可视化
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制text特征点（蓝色）
    ax.scatter(text_2d[:, 0], text_2d[:, 1],
               c='#3498db', alpha=0.7, s=100,
               edgecolors='black', linewidths=1,
               label=f'Text Features (n={len(text_features)})',
               zorder=5)

    # 绘制image特征点（红色）
    ax.scatter(image_2d[:, 0], image_2d[:, 1],
               c='#e74c3c', alpha=0.7, s=100,
               edgecolors='black', linewidths=1,
               label=f'Image Features (n={len(image_features)})',
               zorder=5)

    # 绘制text中心点（大星）
    ax.scatter(text_center[0], text_center[1],
               c='#3498db', s=500, marker='*',
               edgecolors='black', linewidths=2, zorder=10,
               label='Text Center')

    # 绘制image中心点（大星）
    ax.scatter(image_center[0], image_center[1],
               c='#e74c3c', s=500, marker='*',
               edgecolors='black', linewidths=2, zorder=10,
               label='Image Center')

    # 绘制中心点连线
    ax.plot([text_center[0], image_center[0]],
            [text_center[1], image_center[1]],
            'k--', alpha=0.5, linewidth=2)

    # 标注中心点
    ax.annotate('Text', xy=text_center, xytext=(text_center[0], text_center[1] + 2),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.3))
    ax.annotate('Image', xy=image_center, xytext=(image_center[0], image_center[1] - 4),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.3))

    # 添加数据点编号
    for i, (x, y) in enumerate(text_2d):
        ax.text(x, y, str(i+1), fontsize=7, ha='center', va='center',
                color='white', weight='bold', zorder=6)
    for i, (x, y) in enumerate(image_2d):
        ax.text(x, y, str(i+1), fontsize=7, ha='center', va='center',
                color='white', weight='bold', zorder=6)

    # 计算组间距离
    center_distance = np.linalg.norm(text_center - image_center)

    # 设置标题和标签
    ax.set_title(f'{title}\n'
                 f'解释方差: PC1={pca_2d.explained_variance_ratio_[0]*100:.1f}%, '
                 f'PC2={pca_2d.explained_variance_ratio_[1]*100:.1f}% | '
                 f'中心距离: {center_distance:.2f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存至: {save_path}")

    return fig


# ============ 使用示例 ============
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)

    # 创建25个text特征 (1, 768)
    text_features_list = [
        torch.randn(1, 768) for _ in range(25)
    ]

    # 创建5个image特征 (1, 2048)
    image_features_list = [
        torch.randn(1, 2048) for _ in range(5)
    ]

    # 可视化
    fig = visualize_multi_modal_space(
        text_features=text_features_list,
        image_features=image_features_list,
        title='多模态特征空间可视化 (25 Text + 5 Image)',
        save_path='multi_modal_space.png'
    )

    plt.show()
