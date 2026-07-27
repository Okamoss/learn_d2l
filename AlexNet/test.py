import torch
from torch import nn

def evaluate_accuracy_gpu(net, data_loader, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total