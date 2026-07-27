# ==== 评估函数 ====
import torch

def evaluate_accuracy_gpu( data_loader, net, device =None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device :
            device = next(iter(net.parameters())).device
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                correct += (net(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return correct / total

