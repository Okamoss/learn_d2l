import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#=====加载数据函数======
def load_data_fashion_mnist(batch_size):
    #定义transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(root = "./FashionMNISTdata",train=True,
                                                      download = True,transform = transform)
    test_dataset =torchvision.datasets.FashionMNIST(root = "./FashionMNISTdata",train=False,
                                                    download = True,transform = transform)

    train_set = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_set, test_set,len(train_dataset), len(test_dataset)


#======网络: 两个全连接层, 激活函数:ReLu========
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

#初始化全连接层权重
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)

net.apply(init_weights)
#print(net)

#=====超参数======
batch_size= 256
num_epochs = 10
lr= 0.001

# ========损失函数和优化器========
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# ========加载数据========
train_set, test_set,train_size,test_size= load_data_fashion_mnist(batch_size)
print("训练集大小: ", train_size)
print("验证集大小: ",test_size)

writer = SummaryWriter('runs/fashion_mnist')

total_train_step = 0
total_test_step = 0

for epoch in range(num_epochs):
    print(f"------第 {epoch + 1} 轮训练开始------")

    # ======= 开始训练 =======
    for data in train_set:
        imgs, labels = data
        outputs = net(imgs)
        l = loss(outputs, labels)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        total_train_step += 1
        #每一百个样本输出一次loss
        if total_train_step % 100 == 0:
            print(f"训练次数:{total_train_step},Loss:{l.item():.4f}")
            writer.add_scalar("train_loss", l.item(), total_train_step)

# ===== 每个epoch验证一次 =====
    net.eval()
    total_test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data in test_set:
            imgs, labels = data
            outputs = net(imgs)
            l = loss(outputs, labels)
            total_test_loss+=l.item()
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

    print(f"整体验证集上的loss: {total_test_loss:.4f}")
    print(f"整体验证集上的正确率: {total_correct / test_size:.4f}")

    writer.add_scalar("test_loss",  total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_correct / test_size, total_test_step)
    total_test_step += 1

    torch.save(net, f"my_model_{epoch + 1}.pth")
    print(f"model saved as my_model_{epoch + 1}.pth")

writer.close()
print("🎉 训练完成！在终端运行: tensorboard --logdir=runs")




