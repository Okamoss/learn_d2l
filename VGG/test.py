# ======= 评估函数 =======
import torch
from torch import nn


def evaluate_accuracy_gpu(net, data_loader, device =None):
    if isinstance(net, nn.Module):
        net.eval()
        # 如果没有指定设备则 从模型中自动获取设备
        if not device :
            device = next(iter(net.parameters())).device

        correct = 0
        total = 0

        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
            return correct /total




