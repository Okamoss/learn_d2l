import torch
from torch import nn

# ======== 定义VGG块 =========
# num_convs : vgg块里卷积层的数量
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        # kernel_size=3, padding=1, stride=1保持分辨率不变
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        #保持通道数恒定为out_channels
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# ======= 定义自定义网络 ========
# conv_blks = [] 存储每个VGG块
# conv_arch VGG网络架构: 定义每个VGG块的卷积层数量和输出通道数
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1

    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(in_channels*7*7,4096),
        nn.ReLU(),
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096,10)
    )

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


