"""
训练和验证相关函数
"""
from tqdm.auto import tqdm
from .config import BaseConfig as CFG
from .metrics import AvgMeter, get_lr


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        step: "batch" 或 "epoch"

    Returns:
        loss_meter: 损失指标跟踪器
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader):
    """
    验证一个epoch

    Args:
        model: 模型
        valid_loader: 验证数据加载器

    Returns:
        loss_meter: 损失指标跟踪器
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter
