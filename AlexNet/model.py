import netrc

import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            # 第一层: 11x11 卷积, 步幅4
            nn.Conv2d(1,96,11,4,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            # 第二层: 5x5 卷积, padding=2
            nn.Conv2d(96,256,5,1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            #  第三层: 3x3 卷积
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Flatten(),

            #全连接
            nn.Linear(6400,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #nn.Linear(4096,1000),
            nn.Linear(4096,10),
            nn.ReLU(),
        )

#打印台输出可以看每层大小
# net = AlexNet()
# x = torch.randn(1,1,224,224)
# for layer in net.net:
#     x=layer(x)
#     print(layer.__class__.__name__,'Output shape:\t',x.shape)