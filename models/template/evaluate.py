"""
新模型评估脚本
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer
from .config import NewModelConfig
from .model import NewModel
from common import make_train_valid_dfs, build_loaders


def get_image_embeddings(valid_df, model_path, model=None):
    """
    提取验证集的图像嵌入

    Args:
        valid_df: 验证数据DataFrame
        model_path: 模型权重路径
        model: 可选的预加载模型

    Returns:
        model: 模型实例
        image_embeddings: 图像嵌入向量
    """
    config = NewModelConfig()
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    if model is None:
        model = NewModel(config).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    print("提取图像嵌入...")
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, total=len(valid_loader)):
            # 根据你的模型实现提取嵌入
            image_features = model.image_encoder(batch["image"].to(config.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)

    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    """
    根据文本查询检索匹配的图像

    Args:
        model: 模型实例
        image_embeddings: 图像嵌入向量
        query: 查询文本
        image_filenames: 图像文件名列表
        n: 返回的匹配图像数量
    """
    config = NewModelConfig()
    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)
    encoded_query = tokenizer([query])

    batch = {
        key: torch.tensor(values).to(config.device)
        for key, values in encoded_query.items()
    }

    # 编码文本查询
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    # 计算相似度
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    # 获取top-k匹配结果
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    # 可视化结果
    import cv2
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{config.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate(model_path="best_new_model.pt", query="a group of people dancing in a party"):
    """
    评估模型

    Args:
        model_path: 模型权重路径
        query: 查询文本
    """
    print("加载验证数据...")
    _, valid_df = make_train_valid_dfs()

    # 提取图像嵌入
    model, image_embeddings = get_image_embeddings(valid_df, model_path)

    # 执行检索
    print(f"\n查询: {query}")
    print("显示检索结果...")
    find_matches(
        model,
        image_embeddings,
        query=query,
        image_filenames=valid_df['image'].values,
        n=9
    )


if __name__ == "__main__":
    evaluate()
