# learn_d2l

这是我自用的《动手学深度学习》学习代码，主要跟着书和课程手敲练习，没有使用 `d2l` 包。

## 项目说明

- `mlp/`：多层感知机练习。
- `lenet/`：LeNet 练习。
- `lenet_BN/`：加入 BatchNorm 的 LeNet 练习。
- `AlexNet/`：AlexNet 练习。
- `NiN/`：Network in Network 练习。
- `VGG/`：VGG 练习。
- `GoogLeNet/`：GoogLeNet 练习。
- `ResNet/`：ResNet 练习。
- `RNN/`：字符级 RNN 练习。
- `image_augmentation/`：CIFAR-10 数据增强练习。
- `kaggle/`：Kaggle 房价预测代码。
- `gradient/`：梯度消失演示脚本。

## 环境

- Python: 3.12.13
- PyTorch: 2.5.1+cu121
- CUDA 是否可用: True
- CUDA 版本: 12.1

## 数据集说明

由于图像数据集和训练产物体积较大，仓库不会提交以下本地目录：

- `data/`
- `dataset/`
- `FashionMNISTdata/`
- `models/`
- `runs/`

如果需要运行相关训练脚本，运行时会自动下载数据，或者也可以在本地自行准备数据并放到对应目录中。

## 备注

- 这是个人学习记录仓库，代码和目录结构会随着学习过程持续调整。
- 训练脚本会自动下载数据并保存模型权重，方便后续继续实验和对比。
