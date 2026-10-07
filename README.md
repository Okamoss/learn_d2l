# learn_d2l

这是我自己动手整理的《动手学深度学习》PyTorch 学习代码仓库。

项目主要跟着 B 站“土堆”课程练习和《动手学深度学习》内容进行复现，大部分代码没有使用 `d2l` 包，而是尽量自己把数据处理、模型定义、训练流程和保存逻辑手写出来，方便我真正理解每一步在做什么。

## 项目说明

- `mlp/`：多层感知机练习。
- `lenet/`：LeNet 练习。
- `lenet_BN/`：加入 BatchNorm 的 LeNet 练习。
- `AlexNet/`：AlexNet 练习。
- `NiN/`：Network in Network 练习。
- `VGG/`：VGG 练习。
- `GoogLeNet/`：GoogLeNet 练习。
- `ResNet/`：ResNet 练习。
- `SE_ResNet/`：加入 SE 注意力机制的 ResNet 练习。
- `RNN/`：字符级 RNN 练习。
- `GRU/`：GRU 练习。
- `LSTM/`：字符级 LSTM 练习。
- `image_augmentation/`：图像增强练习。
- `gradient/`：梯度消失相关演示代码。
- `kaggle/`：Kaggle 房价预测相关代码。

## 环境

- Python: 3.12.13
- PyTorch: 2.5.1+cu121
- CUDA 是否可用: True
- CUDA 版本: 12.1

## 数据集说明

由于图像数据集和训练产物体积较大，仓库不会提交这些本地目录：

- `data/`
- `dataset/`
- `FashionMNISTdata/`
- `models/`
- `runs/`

运行对应训练脚本时，数据会自动下载；如果某些脚本没有自动下载，也可以手动准备数据后放到对应目录中。

其中，`LSTM/` 里的时间机器文本数据加载目前使用了 `d2l` 的数据接口，运行前需要保证当前环境可以导入 `d2l`。

## 备注

- 这是我的个人学习记录仓库，代码和目录会随着学习过程持续调整。
- 训练脚本通常会自动保存模型权重，方便后续继续实验和对比。
