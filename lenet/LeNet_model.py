from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.net = nn.Sequential(
            # 卷积编码器
            nn.Conv2d(1, 6, 5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            # 全连接层密集块
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.net(x)
        return x
