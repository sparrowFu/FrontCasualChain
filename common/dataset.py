"""
数据集模块 - MSCOCO Captions 数据集
"""
import cv2
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from .config import BaseConfig as CFG


def get_transforms(mode="train", size=224):
    """
    获取图像变换

    Args:
        mode: "train" 或 "valid"/"test"
        size: 图像尺寸

    Returns:
        albumentations 变换组合
    """
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(max_pixel_value=255.0),
    ])


class MSCOCOCaptionsDataset(Dataset):
    """
    MSCOCO Captions 数据集类
    从 parquet 文件加载元数据，从 images/ 目录加载图片
    """

    def __init__(self, dataframe, tokenizer, transforms, image_path):
        """
        Args:
            dataframe: 包含 image_file_name, caption 列的 DataFrame
            tokenizer: 文本 tokenizer
            transforms: 图像变换
            image_path: 图像目录路径
        """
        self.image_filenames = dataframe["image_file_name"].values
        # caption 列是 list[str]，每张图片有多条描述
        self.captions = dataframe["caption"].values
        self.ids = dataframe["original_index"].values
        self.image_path = image_path

        # 1. 展平的 caption 文本列表
        self.flat_captions = []
        
        # 2. 每条 caption 对应的图片索引
        self.caption_to_image_idx = []
        
        # 一次遍历构建映射
        for img_idx, caption_list in enumerate(self.captions):
            for caption in caption_list:
                self.flat_captions.append(caption)
                self.caption_to_image_idx.append(img_idx)

        self.encoded_captions = tokenizer(
            self.flat_captions,
            padding=True,
            truncation=True,
            max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        # 获取编码的文本
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # 获取对应的图片索引
        img_idx = self.caption_to_image_idx[idx]

        # 读取和处理图像
        image_path = os.path.join(self.image_path, self.image_filenames[img_idx])
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        
        # 4. 直接获取 caption（无需索引计算）
        item['caption'] = self.flat_captions[idx]  # 直接获取
        item['id'] = torch.tensor(self.ids[img_idx], dtype=torch.long)

        # # 获取原始 caption（使用 idx 对应的单条 caption）
        # # 重新获取这条 caption
        # caption_idx = idx
        # current_img_idx = 0
        # for caption_list in self.captions:
        #     for _ in caption_list:
        #         if current_img_idx == caption_idx:
        #             item['caption'] = caption_list[caption_idx - sum(len(self.captions[i]) for i in range(img_idx))]
        #             item['id'] = torch.tensor(self.ids[img_idx], dtype=torch.long)
        #             return item
        #         current_img_idx += 1
        #     img_idx += 1

        return item

    def __len__(self):
        return len(self.caption_to_image_idx)


def load_mscoco_data(config: Optional[CFG] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载 MSCOCO Captions 数据集并划分训练集和验证集

    Args:
        config: 配置对象
        test_size: 验证集比例
        random_state: 随机种子

    Returns:
        train_dataframe: 训练数据 DataFrame
        valid_dataframe: 验证数据 DataFrame
    """
    if config is None:
        config = CFG()

    parquet_path = os.path.join(config.captions_path, 'train-00000-of-00001.parquet')

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"找不到 parquet 文件: {parquet_path}")

    # 读取 parquet 文件
    df = pd.read_parquet(parquet_path)

    # 限制数据量（调试模式）
    if config.debug:
        df = df.head(100)

    # 划分训练集和验证集
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    train_df = train_df.reset_index(names="original_index")
    valid_df = valid_df.reset_index(names="original_index")

    return train_df, valid_df


def build_loaders(dataframe, tokenizer, mode, config=None):
    """
    构建数据加载器

    Args:
        dataframe: 包含图像和文本数据的 DataFrame
        tokenizer: 文本 tokenizer
        mode: "train" 或 "valid"/"test"
        config: 配置对象

    Returns:
        dataloader: PyTorch 数据加载器
    """
    if config is None:
        config = CFG()

    transforms = get_transforms(mode=mode, size=config.size)

    dataset = MSCOCOCaptionsDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        transforms=transforms,
        image_path=config.images_path,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
