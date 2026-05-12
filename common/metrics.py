"""
训练指标和工具函数
"""


class AvgMeter:
    """
    平均指标跟踪器 - 用于跟踪训练/验证过程中的指标
    """

    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    """
    获取优化器的当前学习率

    Args:
        optimizer: PyTorch优化器

    Returns:
        当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
