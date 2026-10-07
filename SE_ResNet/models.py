from torch import nn


# ========== 定义 SE 注意力块 ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        weight = self.pool(x)
        weight = weight.reshape(batch_size, channels)
        weight = self.fc(weight)
        weight = weight.reshape(batch_size, channels, 1, 1)

        return x * weight


# ========== 定义带 SE 注意力的残差块 ==========
class SEResidual(nn.Module):
    # strides=1 是整个残差块的默认步长，主要作用于主路径的 3×3 卷积
    # use_1x1conv=True 时，捷径路径用 1×1 卷积调整维度，让两条路径可以相加
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(SEResidual, self).__init__()

        # 主路径
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SEBlock(num_channels)

        # 捷径路径
        # 1×1 卷积调整维度
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.conv3:
            x = self.conv3(x)

        out += x
        out = self.relu(out)
        return out


# ========== 定义 SE-ResNet-18 ==========
class SEResNet18(nn.Module):
    def __init__(self):
        super(SEResNet18, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            SEResidual(64, 64, use_1x1conv=False, strides=1),
            SEResidual(64, 64, use_1x1conv=False, strides=1)
        )

        self.b3 = nn.Sequential(
            SEResidual(64, 128, use_1x1conv=True, strides=2),
            SEResidual(128, 128, use_1x1conv=False, strides=1)
        )

        self.b4 = nn.Sequential(
            SEResidual(128, 256, use_1x1conv=True, strides=2),
            SEResidual(256, 256, use_1x1conv=False, strides=1)
        )

        self.b5 = nn.Sequential(
            SEResidual(256, 512, use_1x1conv=True, strides=2),
            SEResidual(512, 512, use_1x1conv=False, strides=1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
