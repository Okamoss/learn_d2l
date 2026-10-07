import os

import torch
import torchvision
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from SE_ResNet.models import SEResNet18, init_weights
from SE_ResNet.test import evaluate_accuracy_gpu


# ==== 设备配置 ====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==== 数据加载 ====
batch_size = 256
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# === 定义 SE-ResNet 网络 ===
net = SEResNet18()
net.apply(init_weights)

# 打印模型结构
print("SE-ResNet-18 结构:")
X = torch.rand(size=(1, 1, 96, 96))
for name, layer in net.named_children():
    X = layer(X)
    print(f"{name} ({layer.__class__.__name__}): {X.shape}")

# === 超参数 ===
num_epochs = 10
lr = 0.05

# === TensorBoard ===
writer = SummaryWriter("./runs/se_resnet")

# === 训练 ===
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print(f"training on {device}")

best_accuracy = 0
total_step = 0

for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        output = net(X)

        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        total_step += 1

        if total_step % 50 == 0:
            writer.add_scalar("Loss/train", loss.item(), total_step)

    train_accuracy = correct / total_samples
    test_accuracy = evaluate_accuracy_gpu(test_loader, net)

    writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
    writer.add_scalar("Accuracy/test", test_accuracy, epoch + 1)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        os.makedirs("./models", exist_ok=True)
        torch.save(net.state_dict(), "./models/se_resnet")

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

print(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")
