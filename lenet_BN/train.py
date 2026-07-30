import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from lenet_BN.model import LeNet_BN as LeNet
from lenet.LeNet_test import evaluate_accuracy

# ======== 设备配置 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ======= 数据加载 =========
# 使用 Fashion-MNIST 数据集
def load_data_fashion_mnist(batch_size =256, resize = None):
    trans = [transforms.ToTensor()]
    if resize :
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 打印一个样本的 shape
    sample, _ = train_dataset[0]
    print(f"Image shape: {sample.shape}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader,test_loader

# ====== 数据加载 ======
batch_size = 256
train_loader, test_loader = load_data_fashion_mnist(batch_size=batch_size)

# ====== 定义 LeNet模型(LeNet_model) ======

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

# ======= 准确率评估函数 ========

# ====== TensorBoard =======
writer = SummaryWriter("runs/lenet_fashion_mnist")

# ======= 超参数 =======
num_epochs = 10
lr = 0.1
weight_decay = 0

# ======= 创建网络,损失函数和优化器 ========
net = LeNet().to(device)
net.apply(init_weights)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)

# ====== 训练 ========
best_acc = 0.0
total_step = 0

print(f"train on {device}")

for epoch in range(num_epochs):

    total_loss = 0 #每个epoch的所有loss
    batch_num = 0 #每个epoch的batch的数量
    correct = 0 # 每个epoch的正确的数量
    total = 0 # 每个epoch里的样本数

    for data in train_loader:
        X, y = data
        X, y = X.to(device), y.to(device)

        output = net(X)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_num += 1

        # 统计loss
        total_loss += loss.item()
        total_step += 1

        # 统计accuracy
        correct +=  (output.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)

    # 计算当前 epoch 的平均 loss, batch为单位
    train_avg_loss = total_loss / batch_num
    # 计算当前 epoch 的平均 acc
    train_avg_acc = correct / total

# ======测试集评估========
    test_acc = evaluate_accuracy(net, test_loader)

    writer.add_scalar('train_loss', train_avg_loss, epoch)
    writer.add_scalar('train_acc', train_avg_acc, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs("./models", exist_ok=True)
        torch.save(net.state_dict(), "./models/lenet_best.pth")

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {train_avg_loss:.4f},"
        f"Avg Train Acc: {train_avg_acc:.4f}, Test Acc: {test_acc:.4f}")

print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")








