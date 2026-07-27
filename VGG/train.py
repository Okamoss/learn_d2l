import os

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from VGG.model import vgg, init_weights
from VGG.test import evaluate_accuracy_gpu

# ==== 设置设备 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 数据加载 ===
batch_size=128
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== DataLoader ====
train_dataset = torchvision.datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# === 定义VGG 块和网络(model) ===

#==== 超参数 ====
num_epochs = 10
lr = 0.05

# ======= 评估函数(test) =======

# ===== TensorBoard =====
writer = SummaryWriter("./runs/vgg")

# ========= 原始 VGG-11 配置 ========
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

# ======= Fashion-MNIST 用缩小版（通道数 ÷4），避免过拟合 ========
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
print(f"VGG配置: ",small_conv_arch)

# === 训练 ===
net = vgg(small_conv_arch).to(device)
net.apply(init_weights)

optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print(f"training on {device}")

best_acc = 0.0
total_step = 0

for epoch in range(num_epochs):
    net.train()

    correct = 0
    total_loss = 0
    total_samples = 0

    for X, y in train_loader:

        X, y = X.to(device), y.to(device)
        output = net(X)

        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        total_step += 1

        if total_step % 50 == 0:
            writer.add_scalar("Loss_train", loss.item(), total_step)

    train_acc = correct / total_samples
    test_acc = evaluate_accuracy_gpu(net, test_loader)

    writer.add_scalar("train_acc", train_acc, epoch+1)
    writer.add_scalar("test_acc", test_acc, epoch+1)

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs('./models', exist_ok=True)
        torch.save(net.state_dict(), "./models/vgg_best.pth")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, "
          f"Train Accuracy: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f"训练完成! 最佳测试准确集: {best_acc:.4f}")


