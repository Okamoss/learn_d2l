import torch
from torch import nn
from torch.nn import Dropout


# ==== 定义 NiN 块 ====
def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
    )
# ==== 定义 NiN 网络 ====
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature =nn.Sequential(
            nin_block(1,96,11,4,0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(96,256,5,1,2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nin_block(256,384,3,1,1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Dropout(0.5),
            nin_block(384,10,3,1,1),
            #局平均池化层来替代VGG和AlexNet中的全连接层
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
    )

    def forward(self, x):
        return self.feature(x)

