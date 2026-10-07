import os
import math
import torch
from torch import nn

from LSTM.model import LSTM
from lstm_utils import grad_clipping, predict_rnn

# ======== 设备配置 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# ======== 数据加载 (还原书中数据加载方式，但保留在本地) ========
# 注意：这里为了代码独立运行，我稍微封装了一下 d2l 底层的数据加载
# 如果不想用 d2l 原版，可以去手写《时间机器》的读取和分词。
# 这里仍然借用 d2l 的数据加载，因为处理文本非常繁琐，这个和你写的 FashionMNIST 自带下载不同。
try:
    from d2l import torch as d2l
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    print(f"词表大小: {len(vocab)}")
except ImportError:
    print("需要安装 d2l 库才能加载《时间机器》数据集。")
    exit()


# ======== 初始化网络 ========
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


vocab_size = len(vocab)
net = LSTM(vocab_size, num_hiddens=256, num_layers=1).to(device)
net.apply(init_weights)

# ======== 损失函数和优化器 ========
loss_fn = nn.CrossEntropyLoss()
lr, num_epochs = 1.0, 500
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# ======== 训练主循环 ========
print(f"Start training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    net.train()

    total_loss = 0.0
    total_tokens = 0
    state = None  # 每一轮(epoch)开始时重置 State

    for X, Y in train_iter:
        X, Y = X.to(device), Y.to(device)

        # 1. 处理 State 初始化 和 梯度分离
        if state is not None:
            if isinstance(state, tuple):
                for s in state:
                    s.detach_()
            else:
                state.detach_()

        # 2. 前向传播
        y_hat, state = net(X, state)
        y = Y.T.reshape(-1)  # 拉平标签，对齐 y_hat 的维度

        # 3. 计算损失与反向传播
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()

        # 4. 【重要】裁剪梯度，防止 RNN 梯度爆炸
        grad_clipping(net, 1)

        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    # ======= 每个 Epoch 结束后的评估 ========
    ppl = math.exp(total_loss / total_tokens)  # 计算困惑度

    if (epoch + 1) % 20 == 0 or epoch == 0:
        # 生成一段文本检查效果
        pred_str = predict_rnn('time traveller ', 20, net, vocab, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Perplexity: {ppl:.2f}")
        print(f"Prediction: {pred_str}")

print(f"训练完成! 最终困惑度: {ppl:.2f}")