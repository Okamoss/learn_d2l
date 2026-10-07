import torch
from torch import nn


def evaluate_accuracy_gpu(dataloader, net, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = net(X)
            correct += (output.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return correct / total
