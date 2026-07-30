from torch import nn

# ========== 定义残差块 ==========
class Residual(nn.Module):
    #strides=1 是整个残差块的默认步长，主要作用于主路径的 3×3 卷积
    #use_1x1conv=True（即捷径路径需要调整维度），捷径路径的 1×1 卷积也会使用同样的 strides 值，以确保两条路径的输出尺寸一致，能够相加
    def __init__(self, input_channels, num_channels,use_1x1conv=False,strides=1):
        super(Residual, self).__init__()

        #主路径
        self.conv1 = nn.Conv2d(input_channels, num_channels , kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

        # 捷径路径
        # 1*1卷积调整维度
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels , num_channels , kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv3:
            x = self.conv3(x)

        out += x
        out = self.relu(out)
        return out


# ========== 定义 ResNet-18 ==========
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(64, 64, use_1x1conv=False, strides=1),
            Residual(64, 64, use_1x1conv=False, strides=1)
        )

        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, strides=2),
            Residual(128, 128, use_1x1conv=False, strides=1)
        )

        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, strides=2),
            Residual(256, 256, use_1x1conv=False, strides=1)
        )

        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, strides=2),
            Residual(512, 512, use_1x1conv=False, strides=1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
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