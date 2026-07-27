import os

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from AlexNet.model import AlexNet, init_weights
from AlexNet.test import evaluate_accuracy_gpu

# ========= 设备配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: ", device)

# ========= 数据加载 ==========
batch_size =128
# AlexNet 需要 224x224 输入
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data',train= True,transform=transform,download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data',train=False,transform=transform,download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

images, labels = next(iter(train_loader))
print(f"图像批次形状: {images.shape}")

print("训练集大小: ", len(train_dataset))
print("测试集大小: ", len(test_dataset))

# ========== 定义 AlexNet 模型 ==========

# ========= 超参数 ===========
num_epochs = 10
lr=0.01

# ===== TensorBoard =======
writer = SummaryWriter('runs/alexnet')

# ======= 训练 =======
net = AlexNet().to(device)
net.apply(init_weights)

optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

print(f"training model on {device}")

best_acc =0
total_step = 0

for epoch in range(num_epochs):

    net.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_step+=1

        if total_step % 50 == 0:
            writer.add_scalar('train_loss', loss.item(), total_step)

    train_acc = correct / len(train_loader.dataset)
    test_acc = evaluate_accuracy_gpu(net, test_loader)

    if test_acc > best_acc:
        best_acc = test_acc
        os.makedirs('./models', exist_ok=True)
        torch.save(net.state_dict(), './models/alexnet_best.pth')

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)

print(f"训练完成! 最佳测试准确率: {best_acc:.4f}")

