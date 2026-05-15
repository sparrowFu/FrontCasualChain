"""
将 img_ids.txt 中的 img_id 与图片目录中的文件名进行匹配
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_img_ids(file_path: str) -> List[Tuple[int, int]]:
    """
    读取 img_ids.txt 文件
    返回: [(索引, img_id), ...]
    """
    img_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    idx = int(parts[0])
                    img_id = int(parts[1])
                    img_ids.append((idx, img_id))
                else:
                    # 如果没有制表符分隔，整行作为 img_id
                    img_id = int(line)
                    img_ids.append((len(img_ids) + 1, img_id))
    return img_ids


def build_image_lookup(image_dir: str) -> Dict[int, str]:
    """
    构建图片文件名查找表: img_id -> 文件名
    从文件名中解析出 img_id（去除前导零和扩展名）
    """
    lookup = {}
    image_path = Path(image_dir)
    for file in image_path.glob('*.jpg'):
        # 文件名格式: 000000299254.jpg
        name_without_ext = file.stem  # 去除 .jpg
        img_id = int(name_without_ext)  # 转为整数，自动去除前导零
        lookup[img_id] = file.name
    return lookup


def match_img_ids_to_files(img_ids_file: str, image_dir: str) -> Dict:
    """
    匹配 img_id 到图片文件名

    返回字典:
    {
        "matches": [(索引, img_id, 文件名), ...],  # 匹配成功的
        "missing": [(索引, img_id), ...],           # 未找到的
        "match_count": 数量,
        "missing_count": 数量
    }
    """
    # 读取 img_ids
    img_ids = read_img_ids(img_ids_file)

    # 构建图片查找表
    image_lookup = build_image_lookup(image_dir)

    # 匹配
    matches = []
    missing = []

    for idx, img_id in img_ids:
        if img_id in image_lookup:
            matches.append((idx, img_id, image_lookup[img_id]))
        else:
            missing.append((idx, img_id))

    return {
        "matches": matches,
        "missing": missing,
        "match_count": len(matches),
        "missing_count": len(missing),
        "total_count": len(img_ids)
    }


def main():
    # 路径配置
    img_ids_file = r"D:\code\causality\FrontdoorCausalChain\data\mscoco_captions\test\img_ids.txt"
    image_dir = r"D:\code\causality\FrontdoorCausalChain\data\mscoco_captions\images"

    # 执行匹配
    result = match_img_ids_to_files(img_ids_file, image_dir)

    # 输出结果
    print(f"总计: {result['total_count']} 条")
    print(f"匹配成功: {result['match_count']} 条")
    print(f"未找到: {result['missing_count']} 条")
    print()

    # 显示前 10 条匹配结果
    print("=== 匹配示例 (前10条) ===")
    for idx, img_id, filename in result['matches'][:10]:
        print(f"索引 {idx}: img_id={img_id} -> {filename}")

    # 显示未找到的
    if result['missing']:
        print()
        print(f"=== 未找到的图片 ({result['missing_count']}条) ===")
        for idx, img_id in result['missing']:
            print(f"索引 {idx}: img_id={img_id}")

    return result


if __name__ == "__main__":
    result = main()
