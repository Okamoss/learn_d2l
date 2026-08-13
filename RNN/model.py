import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, vocab_size, num_hiddens = 256, num_layers =1):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens #隐藏状态的长度, RNN 每读完一个词，就会生成一个长度为 256 的浮点数数组，用来代表当前时刻的状态,传给下一层
        self.num_layers = num_layers # 每一次RNN的层数

        #可以这样理解:当num_layers=2
        #底层 RNN（第 1 层）：接受X和H,算出一个中间结果
        #高层 RNN:  接收刚才算出来的中间结果, 以及及上一时刻高层的记忆,再进行一次处理得到最终的隐藏状态

        # 1. 输入层：把词元索引转为向量 (Embedding)
        # vocab_size 词表大小
        # num_hiddens（每行的长度，例如 256）。
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # RNN核心层
        #inputsize : embedding把每个词转为num_hiddens长度向量
        # hidden_size: 隐藏状态的特征维度
        # 输入大小（input_size）和隐藏大小（hidden_size）不一定非要相等。我们这里和书里一样相等.
        # 内部真正进行运算的权重矩阵只有 (256, 256) 和 (256, 256),所以不需要知道time step和batchsize
        self.rnn = nn.RNN(input_size =num_hiddens, hidden_size=num_hiddens, num_layers=num_layers)

        # 输出层：把隐藏状态转为词表大小的概率
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, state):
        X = X.permute(1,0)
        X = self.embedding(X)
        output, state = self.rnn(X, state)
        output = output.reshape(-1, output.shape[-1])
        Y = self.fc(output)
        return Y, state

    def begin_state(self, batch_size, device):
        # 形状: (层数, 批量大小, 隐藏单元数)
        # 初始化为全0隐藏状态
        return torch.zeros(self.num_layers, batch_size, self.num_hiddens).to(device)

#理解: X.permute(1, 0)后,手里拿到一个(timestep, batchsize)的矩阵
#      序列1	序列2 序列3
#时间步1	床	 疑	  举
#时间步2	前	 是	  头
#时间步3	明	 地	  望
#时间步4	月	 上	  明
#形状变成了 (4, 3),这 4 就是时间步数
# 预处理会给每个字编号
# [ [0, 4, 5],   <-- 时间步 1：序列1的"床"(0), 序列2的"疑"(4), 序列3的"举"(5)
#   [1, 6, 7],   <-- 时间步 2：序列1的"前"(1), 序列2的"是"(6), 序列3的"头"(7)
#   [2, 8, 9],   <-- 时间步 3：...
#   [3, 10, 2] ] <-- 时间步 4：序列3的"明"是2号

# 经过 Embedding, 在显卡内存里创建了一个巨大的二维字典。
# 28 行 × 256 列。
# 第 0 行：代表数字 0（字“床”）的 256 个特征数字。
# 第 1 行：代表数字 1（字“前”）的 256 个特征数字。
# 以此类推
#可以把这里的256理解成类似通道数量
# 经过 self.embedding 后输出的形状就变成了：(4, 3, 256)。
# output = output.reshape(-1, output.shape[-1]),变成二维的给fc层
#最后的形状(12,vocab_size )
#这是对于一个迭代器的batch的处理,实际上我们一次只处理一个 迭代器batch,state在迭代器batch之间传递状态
#train_iter 默认就是顺序分区的。如果不是顺序分区不用传递state


