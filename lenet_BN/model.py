from torch import nn

class LeNet_BN(nn.Module):
    def __init__(self):
        super(LeNet_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(num_features=6),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(num_features=16),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=16*4*4, out_features=120),
            nn.BatchNorm1d(num_features=120),
            nn.Sigmoid(),

            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(num_features=84),
            nn.Sigmoid(),


            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        return x

net = LeNet_BN()
print(net)

