import os

import torch
import torchvision
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from GoogLeNet.model import net, init_weights
from GoogLeNet.test import evaluate_accuracy

# ==== 设备配置 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# === 数据加载 ===
batch_size = 128
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

#  ==========  超参数 ==========
num_epochs = 10
lr = 0.1  # GoogLeNet 用 0.1

# ==== 网络定义(model) 和 评估函数(test)

# === TensorBoard ===
writer = SummaryWriter("./runs/googlenet")

# === 训练 ===
net = net()
net.apply(init_weights)
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print(net)
print(f"training on device: {device}")

best_acc = 0.0
total_step = 0

for epoch in range(num_epochs):

    net.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        output = net(X)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct = correct + (output.argmax(1) == y).sum()
        total_samples += y.size(0)
        total_step += 1

        if total_step % 50 == 0:
            writer.add_scalar("Loss/train", loss.item(), total_step)

    train_acc = correct / total_samples
    writer.add_scalar("Accuracy/train", train_acc, epoch+1)
    test_acc = evaluate_accuracy( test_loader,net)
    writer.add_scalar("Accuracy/test", test_acc, epoch+1)

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("./models",exist_ok=True)
        torch.save(net.state_dict(),"./models/googlenet.pth")

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")




