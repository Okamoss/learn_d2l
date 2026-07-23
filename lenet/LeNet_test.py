import torch
from torch import nn

def evaluate_accuracy(net,data_loader,device = None):
    if isinstance(net,nn.Module):
        net.eval()
        # 自动检测并设置模型运行的设备
        if not device:
            device = next(iter(net.parameters())).device

    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data
            X, y = X.to(device), y.to(device)
            output = net(X)
            # 样本里正确是数量
            correct += (output.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
    return correct / total