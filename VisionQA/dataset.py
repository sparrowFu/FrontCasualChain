"""
VisionQA 数据集模块
"""
import os
import torch
import cv2
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import albumentations as A
# from albumentations import A
from typing import Optional, List, Dict
from .config import VQAConfig


def get_vqa_transforms(mode="train", size=224):
    """
    获取 VQA 图像变换

    Args:
        mode: "train" 或 "test"
        size: 图像尺寸

    Returns:
        albumentations 变换组合
    """
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(max_pixel_value=255.0),
    ])


def read_vqa_file(file_path: str) -> List[str]:
    """
    读取 VQA 数据文件

    Args:
        file_path: 文件路径

    Returns:
        内容列表（去除序号）
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 分割序号和内容
                parts = line.split('\t')
                if len(parts) == 2:
                    data.append(parts[1])
                else:
                    # 尝试按空格分割
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        data.append(parts[1])
                    else:
                        data.append(line)
    return data


def read_vqa_ids(file_path: str) -> List[int]:
    """
    读取 VQA 图片 ID 文件

    Args:
        file_path: 文件路径

    Returns:
        ID 列表（整数）
    """
    ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    ids.append(int(parts[1]))
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        ids.append(int(parts[1]))
                    else:
                        ids.append(int(line))
    return ids


def read_vqa_types(file_path: str) -> List[int]:
    """
    读取 VQA 问题类型文件

    Args:
        file_path: 文件路径

    Returns:
        类型列表（整数）
    """
    types = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    types.append(int(parts[1]))
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        types.append(int(parts[1]))
                    else:
                        types.append(int(line))
    return types


class VQADataset(Dataset):
    """
    VisionQA 数据集类
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: DistilBertTokenizer,
        transforms,
        image_path: str,
        filtered: bool = False
    ):
        """
        Args:
            data_dir: 数据目录（包含 img_ids.txt, questions.txt, answers.txt, types.txt, img_filenames.txt）
            tokenizer: 文本 tokenizer
            transforms: 图像变换
            image_path: 图像目录路径
            filtered: 是否使用过滤后的数据（测试集）
        """
        self.data_dir = data_dir
        self.image_path = image_path
        self.transforms = transforms

        # 读取数据文件
        suffix = "_filtered" if filtered else ""

        self.img_ids = read_vqa_ids(os.path.join(data_dir, f"img_ids{suffix}.txt"))
        self.img_filenames = read_vqa_file(os.path.join(data_dir, f"img_filenames{suffix}.txt"))
        self.questions = read_vqa_file(os.path.join(data_dir, f"questions{suffix}.txt"))
        self.answers = read_vqa_file(os.path.join(data_dir, f"answers{suffix}.txt"))
        self.types = read_vqa_types(os.path.join(data_dir, f"types{suffix}.txt"))

        # 验证数据长度一致
        assert len(self.img_ids) == len(self.questions) == len(self.answers) == len(self.types) == len(self.img_filenames), \
            f"数据文件长度不一致: ids={len(self.img_ids)}, questions={len(self.questions)}, answers={len(self.answers)}, types={len(self.types)}, filenames={len(self.img_filenames)}"

        # 构建答案到索引的映射
        self.answer2idx = {}
        self.idx2answer = []
        self.answer_indices = []

        for answer in self.answers:
            if answer not in self.answer2idx:
                self.answer2idx[answer] = len(self.idx2answer)
                self.idx2answer.append(answer)
            self.answer_indices.append(self.answer2idx[answer])

        self.num_answers = len(self.idx2answer)

        # 编码问题
        self.encoded_questions = tokenizer(
            self.questions,
            padding=True,
            truncation=True,
            max_length=200
        )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            dict: {
                'image': tensor,
                'input_ids': tensor,
                'attention_mask': tensor,
                'answer_idx': int,
                'question_type': int,
                'question': str,
                'answer': str,
                'img_filename': str
            }
        """
        # 获取编码的问题
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_questions.items()
        }

        # 读取图像
        img_filename = self.img_filenames[idx]
        image_path = os.path.join(self.image_path, img_filename)
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()

        # 其他信息
        item['answer_idx'] = torch.tensor(self.answer_indices[idx], dtype=torch.long)
        item['question_type'] = torch.tensor(self.types[idx], dtype=torch.long)
        item['question'] = self.questions[idx]
        item['answer'] = self.answers[idx]
        item['img_filename'] = img_filename

        return item

    def get_answer_vocab_size(self):
        """获取答案词汇表大小"""
        return self.num_answers


def build_vqa_loaders(
    train_dir: str,
    test_dir: str,
    tokenizer: DistilBertTokenizer,
    image_path: str,
    config: Optional[VQAConfig] = None
):
    """
    构建 VQA 数据加载器

    Args:
        train_dir: 训练数据目录
        test_dir: 测试数据目录
        tokenizer: 文本 tokenizer
        image_path: 图像目录路径
        config: 配置对象

    Returns:
        train_loader, test_loader, train_dataset
    """
    if config is None:
        config = VQAConfig()

    transforms = get_vqa_transforms(mode="train", size=config.size)

    # 训练集
    train_dataset = VQADataset(
        data_dir=train_dir,
        tokenizer=tokenizer,
        transforms=transforms,
        image_path=image_path,
        filtered=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    # 测试集（使用过滤后的文件）
    test_transforms = get_vqa_transforms(mode="test", size=config.size)
    test_dataset = VQADataset(
        data_dir=test_dir,
        tokenizer=tokenizer,
        transforms=test_transforms,
        image_path=image_path,
        filtered=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    return train_loader, test_loader, train_dataset
