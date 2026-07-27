# 按照lr =0.1效果比较差,无法收敛
# 这里学习率修改成0.01,并优化器加入了动量

import os

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from NiN.model import Net
from NiN.model import init_weights
from NiN.test import evaluate_accuracy_gpu

# ===== 设备配置 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device",device)

# ==== 数据加载 ====
batch_size = 128

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

print(f"训练集大小: ",len(train_dataset))
print(f"测试集大小: ",len(test_dataset))

# ==== 定义NiN网络(model.py) ======

# === 朝参宿 ===
num_epochs = 10
lr=0.01 #设置成0.1效果比较差?

# ==== 准确率评估函数(test) ====

# ==== TensorBoard ====
writer = SummaryWriter("./runs/nin")

# ==== 训练 ====
net = Net()
net.apply(init_weights)
#print(net)
net.to(device)

# 加动量不然结果较差
optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

print(f"training on {device}")

best_acc = 0
total_step = 0

for epoch in range(num_epochs):
    net.train()
    correct = 0
    total_loss = 0
    total_sample = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = net(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_sample += y.size(0)
        correct += (output.argmax(dim=1) == y).sum().item()
        total_step += 1

        if total_step % 50 == 0:
            writer.add_scalar('Loss/train', loss.item(), total_step)

    train_acc = correct / total_sample
    test_acc = evaluate_accuracy_gpu(test_loader,net)

    writer.add_scalar('Train_acc', train_acc, epoch+1)
    writer.add_scalar('Test_acc', test_acc, epoch+1)

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("./models",exist_ok=True)
        torch.save(net.state_dict(),"./models/nin.pth")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f},"
          f"Train Acc: {train_acc:.4f},Test Acc: {test_acc:.4f}")

print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")



