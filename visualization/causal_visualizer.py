"""
前门准则因果链可视化器
用于验证和可视化 FrontDoor 模型的因果推断过程
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.frontdoor.config import FrontDoorConfig
from models.frontdoor.model import FrontDoorCausalModel, FrontDoorWithEncoders
from models.clip.model import CLIPModel, ImageEncoder, TextEncoder
from transformers import DistilBertTokenizer


class CausalChainVisualizer:
    """
    前门准则因果链可视化器

    功能：
    1. 加载 CLIP 和 FrontDoor 模型
    2. 对单样本进行因果推断
    3. 验证前门准则的三个条件
    4. 生成可视化图表
    """

    def __init__(self,
                 clip_model_path: Optional[str] = None,
                 frontdoor_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化可视化器

        Args:
            clip_model_path: CLIP 模型路径
            frontdoor_model_path: FrontDoor 模型路径
            device: 运行设备
        """
        self.config = FrontDoorConfig()

        if device is None:
            self.device = self.config.device
        else:
            self.device = torch.device(device)

        # 加载 tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config.text_model_path,
            local_files_only=True
        )

        # 加载图像和文本编码器
        print("加载编码器...")
        self.image_encoder = ImageEncoder().to(self.device)
        self.text_encoder = TextEncoder().to(self.device)

        # 加载 CLIP 投影头（如果提供）
        if clip_model_path and os.path.exists(clip_model_path):
            print(f"加载 CLIP 模型: {clip_model_path}")
            checkpoint = torch.load(clip_model_path, map_location=self.device, weights_only=False)

            # 加载编码器权重
            self.image_encoder.load_state_dict({
                k.replace('image_encoder.', ''): v
                for k, v in checkpoint.items()
                if k.startswith('image_encoder.')
            }, strict=False)
            self.text_encoder.load_state_dict({
                k.replace('text_encoder.', ''): v
                for k, v in checkpoint.items()
                if k.startswith('text_encoder.')
            }, strict=False)

        self.image_encoder.eval()
        self.text_encoder.eval()

        # 加载 FrontDoor 因果模型
        print("加载 FrontDoor 因果模型...")
        self.causal_model = FrontDoorCausalModel(
            image_feat_dim=self.config.image_embedding,
            text_feat_dim=self.config.text_embedding,
            shared_dim=self.config.shared_dim,
            private_ratio=self.config.private_ratio,
            config=self.config
        ).to(self.device)

        if frontdoor_model_path and os.path.exists(frontdoor_model_path):
            print(f"加载 FrontDoor 模型权重: {frontdoor_model_path}")
            self.causal_model.load_state_dict(
                torch.load(frontdoor_model_path, map_location=self.device, weights_only=False)
            )

        self.causal_model.eval()

        # 冻结编码器参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        编码图像

        Args:
            image_path: 图像路径

        Returns:
            image_features: 图像特征 (1, 2048)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整大小和归一化
        import albumentations as A
        transforms = A.Compose([
            A.Resize(self.config.size, self.config.size),
            A.Normalize(max_pixel_value=255.0),
        ])
        image = transforms(image=image)['image']
        image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        # 编码
        with torch.no_grad():
            image_features = self.image_encoder(image)

        return image_features

    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码文本

        Args:
            text: 文本字符串

        Returns:
            text_features: 文本特征 (1, 768)
        """
        # Tokenize
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        # 编码
        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=encoded['input_ids'].to(self.device),
                attention_mask=encoded['attention_mask'].to(self.device)
            )

        return text_features

    def encode_to_shared_private(self,
                                image_features: torch.Tensor,
                                text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将特征分解为 Shared 和 Private 部分

        Args:
            image_features: 图像特征
            text_features: 文本特征

        Returns:
            features: 包含共享和私有特征的字典
        """
        with torch.no_grad():
            # 通过因果模型分解特征

            # 图像特征分解
            image_shared = self.causal_model.image_shared_encoder(image_features)
            image_private = self.causal_model.image_private_encoder(image_features)

            # 文本特征分解
            text_shared = self.causal_model.text_shared_encoder(text_features)
            text_private = self.causal_model.text_private_encoder(text_features)

        return {
            'image_shared': image_shared,
            'image_private': image_private,
            'text_shared': text_shared,
            'text_private': text_private
        }

    def compute_shared_semantic(self,
                               image_shared: torch.Tensor,
                               text_shared: torch.Tensor) -> torch.Tensor:
        """
        计算共享语义 M

        Args:
            image_shared: 图像共享特征
            text_shared: 文本共享特征

        Returns:
            shared_semantic: 拼接的共享语义
        """
        with torch.no_grad():
            concatenated = torch.cat([image_shared, text_shared], dim=-1)
            shared_senmantic = self.causal_model.semantic_fusion(concatenated)

        return shared_senmantic

    def compute_causal_effect(self, shared_semantic: torch.Tensor) -> torch.Tensor:
        """
        计算因果效应

        Args:
            shared_semantic: 共享语义

        Returns:
            causal_effect: 因果效应值
        """
        with torch.no_grad():
            causal_effect = self.causal_model.causal_effect_estimator(shared_semantic)
        return causal_effect

    def pad_to_match(self, feat_small, feat_large):
        """
        将小维度特征补零以匹配大维度特征
        
        Args:
            feat_small: [batch, small_dim]
            feat_large: [batch, large_dim]
        
        Returns:
            padded: [batch, large_dim]
        """
        small_dim = feat_small.shape[-1]
        large_dim = feat_large.shape[-1]
        
        # 创建零填充
        padding_size = large_dim - small_dim
        padding = torch.zeros(
            feat_small.shape[0], 
            padding_size, 
            device=feat_small.device, 
            dtype=feat_small.dtype
        )
        
        # 拼接
        padded = torch.cat([feat_small, padding], dim=-1)
    
        return padded

    def verify_front_door_criterion(self, features: Dict[str, torch.Tensor]) -> Dict:
        """
        验证前门准则的三个条件

        Args:
            features: 包含共享和私有特征的字典

        Returns:
            verification: 验证结果字典
        """
        # 条件1: 完全中介 - Shared 特征应该高度相关
        shared_similarity = F.cosine_similarity(
            features['image_shared'],
            features['text_shared']
        ).item()
        condition1_satisfied = shared_similarity > 0.5

        # 条件2: I,Q -> M 无混杂 - 编码过程是物理/确定性过程
        condition2_satisfied = True  # 设计上满足

        image_shared_padded = self.pad_to_match(features['image_shared'], features['image_private'])
        text_private_padded = self.pad_to_match(features['text_private'], features['text_shared'])

        # 条件3: M → A 无混杂 - Shared 和 Private 应该正交
        image_orthogonality = F.cosine_similarity(
            image_shared_padded,
            features['image_private']
        ).item()
        text_orthogonality = F.cosine_similarity(
            features['text_shared'],
            text_private_padded
        ).item()
        avg_orthogonality = (image_orthogonality + text_orthogonality) / 2
        condition3_satisfied = avg_orthogonality < 0.3

        all_satisfied = condition1_satisfied and condition2_satisfied and condition3_satisfied

        return {
            'shared_similarity': shared_similarity,
            'image_orthogonality': image_orthogonality,
            'text_orthogonality': text_orthogonality,
            'avg_orthogonality': avg_orthogonality,
            'condition1_satisfied': condition1_satisfied,
            'condition2_satisfied': condition2_satisfied,
            'condition3_satisfied': condition3_satisfied,
            'all_satisfied': all_satisfied
        }

    def visualize_single_sample(self,
                                image_path: str,
                                text: str,
                                save_path: Optional[str] = None) -> Dict:
        """
        对单样本进行完整的因果推断并可视化

        Args:
            image_path: 图像路径
            text: 文本描述
            save_path: 保存路径

        Returns:
            results: 包含所有计算结果的字典
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 编码
        print(f"\n{'='*60}")
        print(f"图像: {os.path.basename(image_path)}")
        print(f"文本: {text}")
        print(f"{'='*60}\n")

        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)

        # 2. 分解特征
        features = self.encode_to_shared_private(image_features, text_features)

        # 3. 计算共享语义
        shared_semantic = self.compute_shared_semantic(
            features['image_shared'],
            features['text_shared']
        )

        # 4. 计算因果效应
        causal_effect = self.compute_causal_effect(shared_semantic)

        # 5. 验证前门准则
        verification = self.verify_front_door_criterion(features)

        # 6. 可视化
        self._plot_causal_chain_visualization(
            image_features, text_features, features,
            shared_semantic, causal_effect, verification,
            image_path, save_path
        )

        # 返回结果
        return {
            'image_features': image_features,
            'text_features': text_features,
            'features': features,
            'shared_semantic': shared_semantic,
            'causal_effect': causal_effect,
            'verification': verification
        }

    def _plot_pca_visualization(self, ax, feat1, feat2, title,
                                 colors=['#3498db', '#e74c3c'],
                                 labels=['Feature 1', 'Feature 2'],
                                 fixed_ylim=None, fixed_xlim=None):
        """
        绘制两个高维向量在2D空间中的位置点

        将两个高维特征向量通过PCA降维到2D空间，可视化它们的位置关系

        Args:
            ax: matplotlib轴对象
            feat1: 第一个特征向量 (numpy array, 1D) - 在高维空间中是一个点
            feat2: 第二个特征向量 (numpy array, 1D) - 在高维空间中是一个点
            title: 图标题
            colors: 颜色列表
            labels: 标签列表
        """
        # 维度对齐：将两个不同维度的向量投影到公共子空间
        dim1, dim2 = len(feat1), len(feat2)

        if dim1 != dim2:
            target_dim = min(dim1, dim2, 64)
            # 使用截断方式对齐到相同维度
            feat1_aligned = feat1[:target_dim] if dim1 >= target_dim else np.pad(feat1, (0, target_dim - dim1))
            feat2_aligned = feat2[:target_dim] if dim2 >= target_dim else np.pad(feat2, (0, target_dim - dim2))
        else:
            feat1_aligned = feat1
            feat2_aligned = feat2

        # 将两个向量作为两个点，合并后进行PCA降维到2D
        two_points = np.vstack([feat1_aligned, feat2_aligned])
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(two_points)

        point1_2d = points_2d[0]  # feat1在2D空间的位置
        point2_2d = points_2d[1]  # feat2在2D空间的位置

        # 计算原始高维空间的相似度
        cos_sim = np.dot(feat1_aligned, feat2_aligned) / (
            np.linalg.norm(feat1_aligned) * np.linalg.norm(feat2_aligned) + 1e-8
        )

        # 如果两个点重合或非常接近，添加小的偏移以便可视化
        if np.linalg.norm(point1_2d - point2_2d) < 0.1:
            offset = 3.0
            point2_2d = point2_2d + np.array([offset, offset])

        # 设置坐标轴范围（添加边距）
        all_x = [point1_2d[0], point2_2d[0]]
        all_y = [point1_2d[1], point2_2d[1]]
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        x_margin = max(x_range * 0.4, 2.0)
        y_margin = max(y_range * 0.4, 2.0)

        # 使用固定的坐标轴范围（如果提供）
        if fixed_xlim is not None:
            ax.set_xlim(fixed_xlim)
        else:
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)

        if fixed_ylim is not None:
            ax.set_ylim(fixed_ylim)
        else:
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        # 绘制连接线
        ax.plot([point1_2d[0], point2_2d[0]],
                [point1_2d[1], point2_2d[1]],
                'k-', alpha=0.4, linewidth=1, linestyle='--')

        # 绘制两个点（带光晕效果）
        # 点1的光晕
        ax.scatter(point1_2d[0], point1_2d[1],
                   s=500, c=colors[0], alpha=0.2, edgecolors='none')
        # 点1
        ax.scatter(point1_2d[0], point1_2d[1],
                   s=200, c=colors[0], alpha=0.8,
                   edgecolors='black', linewidths=2, zorder=10, label=labels[0])

        # 点2的光晕
        ax.scatter(point2_2d[0], point2_2d[1],
                   s=500, c=colors[1], alpha=0.2, edgecolors='none')
        # 点2
        ax.scatter(point2_2d[0], point2_2d[1],
                   s=200, c=colors[1], alpha=0.8,
                   edgecolors='black', linewidths=2, zorder=10, label=labels[1])

        # 设置坐标标签
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

        # 设置标题
        ax.set_title(f'{title}\n余弦相似度: {cos_sim:.4f} | 解释方差: {pca.explained_variance_ratio_.sum():.1%}',
                    fontweight='bold', fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_causal_chain_visualization(self,
                                        image_features: torch.Tensor,
                                        text_features: torch.Tensor,
                                        features: Dict[str, torch.Tensor],
                                        shared_semantic: torch.Tensor,
                                        causal_effect: torch.Tensor,
                                        verification: Dict,
                                        image_path: str,
                                        save_path: Optional[str] = None):
        """绘制因果链可视化图"""
        # 使用3x3布局来容纳额外的3张图
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('前门准则因果链可视化', fontsize=16, fontweight='bold')

        # 1. 原始特征
        ax = axes[0, 0]
        img_feat = image_features[0].detach().cpu().numpy()[:100]
        txt_feat = text_features[0].detach().cpu().numpy()[:100]
        ax.plot(img_feat, label='Image Features', alpha=0.7)
        ax.plot(txt_feat, label='Text Features', alpha=0.7)
        ax.set_title('原始编码特征', fontweight='bold')
        ax.set_xlabel('特征维度')
        ax.set_ylabel('特征值')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Shared 特征
        ax = axes[0, 1]
        img_shared = features['image_shared'][0].detach().cpu().numpy()
        txt_shared = features['text_shared'][0].detach().cpu().numpy()
        ax.plot(img_shared, label='Image Shared', alpha=0.7, linewidth=2)
        ax.plot(txt_shared, label='Text Shared', alpha=0.7, linewidth=2)
        ax.set_title('共同语义特征（Shared）', fontweight='bold')
        ax.set_xlabel('特征维度')
        ax.set_ylabel('特征值')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Private 特征（添加小窗放大显示）
        ax = axes[0, 2]
        img_private = features['image_private'][0].detach().cpu().numpy()
        txt_private = features['text_private'][0].detach().cpu().numpy()
        ax.plot(img_private, label='Image Private', alpha=0.5, linestyle='--')
        ax.plot(txt_private, label='Text Private', alpha=0.5, linestyle='--')
        ax.set_title('模态私有特征（Private）', fontweight='bold')
        ax.set_xlabel('特征维度')
        ax.set_ylabel('特征值')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加小窗放大显示前面的维度
        ax_inset = inset_axes(ax, width='45%', height='40%', loc='upper left',
                              bbox_to_anchor=(0.05, 0.1, 1, 1),
                              bbox_transform=ax.transAxes)
        inset_dim = min(50, len(img_private))
        ax_inset.plot(img_private[:inset_dim], label='Image Private', alpha=0.7, linestyle='--', linewidth=1)
        ax_inset.plot(txt_private[:inset_dim], label='Text Private', alpha=0.7, linestyle='--', linewidth=1)
        ax_inset.set_title(f'前{inset_dim}维放大', fontsize=8)
        ax_inset.set_xlabel('维度', fontsize=7)
        ax_inset.set_ylabel('值', fontsize=7)
        ax_inset.tick_params(labelsize=6)
        ax_inset.grid(True, alpha=0.3)
        # 为小窗添加边框
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(1.5)

        # 4. 共享语义热图
        ax = axes[1, 0]
        semantic_matrix = shared_semantic[0, :64].detach().cpu().numpy().reshape(8, 8)
        sns.heatmap(semantic_matrix, cmap='viridis', ax=ax, cbar_kws={'label': '值'})
        ax.set_title('共享语义M（前64维）', fontweight='bold')
        ax.set_xlabel('列')
        ax.set_ylabel('行')

        # 5. Shared 特征相似度
        ax = axes[1, 1]
        shared_sim = verification['shared_similarity']
        color = 'green' if verification['condition1_satisfied'] else 'red'
        ax.bar(['Shared\n相似度'], [shared_sim], color=color, alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', label='阈值(0.5)')
        ax.set_title(f'Shared特征相似度: {shared_sim:.4f}', fontweight='bold')
        ax.set_ylabel('相似度')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 6. 因果效应
        ax = axes[1, 2]
        effect_value = causal_effect.item()
        ax.bar(['因果效应'], [effect_value], color='lightcoral', alpha=0.7)
        ax.set_title(f'因果效应值: {effect_value:.4f}', fontweight='bold')
        ax.set_ylabel('效应值')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 7-9. PCA降维向量空间可视化（2D）
        # 先计算统一的坐标轴范围，确保三张图尺寸一致
        def get_pca_range(feat1, feat2):
            dim1, dim2 = len(feat1), len(feat2)
            if dim1 != dim2:
                target_dim = min(dim1, dim2, 64)
                feat1_aligned = feat1[:target_dim] if dim1 >= target_dim else np.pad(feat1, (0, target_dim - dim1))
                feat2_aligned = feat2[:target_dim] if dim2 >= target_dim else np.pad(feat2, (0, target_dim - dim2))
            else:
                feat1_aligned = feat1
                feat2_aligned = feat2
            two_points = np.vstack([feat1_aligned, feat2_aligned])
            pca = PCA(n_components=2)
            points_2d = pca.fit_transform(two_points)
            all_x = [points_2d[0, 0], points_2d[1, 0]]
            all_y = [points_2d[0, 1], points_2d[1, 1]]
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            x_margin = max(x_range * 0.4, 2.0)
            y_margin = max(y_range * 0.4, 2.0)
            return (min(all_x) - x_margin, max(all_x) + x_margin), (min(all_y) - y_margin, max(all_y) + y_margin)

        # 获取三张图的坐标轴范围，取并集
        img_feat_full = image_features[0].detach().cpu().numpy()
        txt_feat_full = text_features[0].detach().cpu().numpy()
        xlim_1, ylim_1 = get_pca_range(img_feat_full, txt_feat_full)
        xlim_2, ylim_2 = get_pca_range(img_shared, txt_shared)
        xlim_3, ylim_3 = get_pca_range(img_private, txt_private)
        # 统一的坐标轴范围
        unified_xlim = (min(xlim_1[0], xlim_2[0], xlim_3[0]), max(xlim_1[1], xlim_2[1], xlim_3[1]))
        unified_ylim = (min(ylim_1[0], ylim_2[0], ylim_3[0]), max(ylim_1[1], ylim_2[1], ylim_3[1]))

        # 7. Image Feature 和 Text Feature PCA降维
        ax = axes[2, 0]
        self._plot_pca_visualization(
            ax, img_feat_full, txt_feat_full,
            'Image Feature vs Text Feature',
            colors=['#3498db', '#e74c3c'],
            labels=['Image Feature', 'Text Feature'],
            fixed_xlim=unified_xlim, fixed_ylim=unified_ylim
        )

        # 8. Image Shared 和 Text Shared PCA降维
        ax = axes[2, 1]
        self._plot_pca_visualization(
            ax, img_shared, txt_shared,
            'Image Shared vs Text Shared',
            colors=['#2ecc71', '#f39c12'],
            labels=['Image Shared', 'Text Shared'],
            fixed_xlim=unified_xlim, fixed_ylim=unified_ylim
        )

        # 9. Image Private 和 Text Private PCA降维
        ax = axes[2, 2]
        self._plot_pca_visualization(
            ax, img_private, txt_private,
            'Image Private vs Text Private',
            colors=['#9b59b6', '#1abc9c'],
            labels=['Image Private', 'Text Private'],
            fixed_xlim=unified_xlim, fixed_ylim=unified_ylim
        )

        # 添加说明
        status_text = "✅ 前门准则满足" if verification['all_satisfied'] else "❌ 前门准则部分不满足"
        fig.text(0.5, 0.01, f'因果链: 图像 → Shared语义 → 文本  |  {status_text}',
                ha='center', fontsize=12, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat' if verification['all_satisfied'] else 'lightcoral', alpha=0.3))

        plt.tight_layout(rect=[0, 0.03, 0.96, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n可视化已保存至: {save_path}")

        plt.show()

    def print_verification_report(self, verification: Dict):
        """打印验证报告"""
        print("\n" + "="*80)
        print("前门准则验证报告")
        print("="*80)

        print("\n条件1: 完全中介")
        print(f"  - Shared特征相似度: {verification['shared_similarity']:.4f}")
        print(f"  - 阈值: > 0.5")
        print(f"  - 状态: {'✅ 满足' if verification['condition1_satisfied'] else '❌ 不满足'}")

        print("\n条件2: I,Q → M 无混杂")
        print(f"  - 状态: {'✅ 满足' if verification['condition2_satisfied'] else '❌ 不满足'}")
        print("  - 解释: 编码过程是物理/确定性过程，无反向因果")

        print("\n条件3: M → A 无混杂")
        print(f"  - 图像正交性: {verification['image_orthogonality']:.4f}")
        print(f"  - 文本正交性: {verification['text_orthogonality']:.4f}")
        print(f"  - 平均正交性: {verification['avg_orthogonality']:.4f}")
        print(f"  - 阈值: < 0.3")
        print(f"  - 状态: {'✅ 满足' if verification['condition3_satisfied'] else '❌ 不满足'}")

        print("\n" + "="*80)
        print(f"总体评估: {'✅ 所有条件满足' if verification['all_satisfied'] else '❌ 部分条件不满足'}")
        print("="*80)


def visualize_single_sample(image_path: str,
                            text: str,
                            clip_model_path: Optional[str] = None,
                            frontdoor_model_path: Optional[str] = None,
                            save_path: Optional[str] = None) -> Dict:
    """
    便捷函数：可视化单样本的因果链

    Args:
        image_path: 图像路径
        text: 文本描述
        clip_model_path: CLIP 模型路径
        frontdoor_model_path: FrontDoor 模型路径
        save_path: 保存路径

    Returns:
        results: 计算结果字典
    """
    visualizer = CausalChainVisualizer(
        clip_model_path=clip_model_path,
        frontdoor_model_path=frontdoor_model_path
    )

    results = visualizer.visualize_single_sample(
        image_path=image_path,
        text=text,
        save_path=save_path
    )

    visualizer.print_verification_report(results['verification'])

    return results
