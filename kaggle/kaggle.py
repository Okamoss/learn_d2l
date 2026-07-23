import hashlib
import os
import tarfile
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

#下载方式先当黑箱
#   ======="""下载一个DATA_HUB中的文件，返回本地文件名"""==========
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

# =========== """下载并解压zip/tar文件"""============
def download_extract(name, folder=None):  #@save
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

#====== """下载DATA_HUB中的所有文件"""=======
def download_all():
    for name in DATA_HUB:
        download(name)

# =========设置数据集字典=========
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

# ==========  加载数据 ==========
print("下载数据中...")
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(f"训练集: {train_data.shape}")   # 应该是 (1460, 81)
print(f"测试集: {test_data.shape}")     # 应该是 (1459, 80)


#print(train_data.head())
#print(test_data.head())

# ========== 数据预处理 ===========
print("\n数据预处理...")

#合并训练集和测试集
all_features = pd.concat((train_data.iloc[:,1:-1], test_data.iloc[:,1:-1]))
#print(all_features.head())

#处理数据特征: 标准化 + 填充缺失数据

# 第1步：查看所有列的数据类型
dtypes = all_features.dtypes
# 第2步：判断哪些列不是文本（object）
#is_numeric = dtypes != 'object'(沐神源代码可能保留str)
is_numeric = (dtypes != 'object') & (dtypes != 'str')
#print(dtypes)
#print(is_numeric)
# 第3步：用这个判断去筛选，只保留数值类型的列(保留true)
numeric_dtypes = dtypes[is_numeric]
#print(numeric_dtypes)
# 第4步：取这些列的列名
numeric_features = numeric_dtypes.index
# 现在 numeric_features 就是所有数值列的列名列表
#print(numeric_features)

#标准化: (原始值 - 均值) / 标准差
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
#填充缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
#print(all_features.head())

#处理类别特征: one-hot编码
all_features = pd.get_dummies(all_features,dummy_na=True)
#print(all_features.head())
print(f"特征工程后特征数: {all_features.shape[1]}")

#转化为tensor类型
n_train = train_data.shape[0] #训练行的行数

#把所有类型型转化为浮点类型,主要原因是 dtype=torch.float32 无法转化bool类型
all_features = all_features.astype(np.float32)

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_feature = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)

print(f"训练特征: {train_features.shape}, 训练标签: {train_labels.shape}")
print(f"测试特征: {test_feature.shape}")


# ========== 设备配置 ===========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\ndevice: {device}")

# ========= 定义模型 ===========
class HousePriceModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight,nonlinearity = 'relu')

# ========== 创建网路 ==========
in_features = train_features.shape[1] #特征(列)数量
net = HousePriceModel(in_features).to(device)
net.apply(init_weights)

# ============= 超参数 =============
batch_size = 64
num_epochs = 100
lr = 0.01
weight_decay = 0.01

# ========== 损失函数和优化器 ===========
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

#  ===========创建DataLoader(无验证集) ===========
dataset_train = TensorDataset(train_features, train_labels)
dataset_test = TensorDataset(test_feature)
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size,shuffle=False)

# ========== 定于 Log RMSE ============
# 定义均方根误差,用于验证,先作为黑箱
def log_rmse(preds, labels):
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    return torch.sqrt(loss_fn(torch.log(clipped_preds), torch.log(labels))).item()

# ======== TensorBoard =======
writer = SummaryWriter('runs/house_price1')

# =========== 训练 ==========
print("\n开始训练")
total_train_step = 0
best_rmse = float('inf')  # 初始无穷大，找最小的 RMSE

for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    total_rmse = 0
    num_batches = 0

    for X, y in train_loader:
        X,y =X.to(device), y.to(device)

        output = net(X)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += log_rmse(output, y)
        num_batches += 1

        total_train_step += 1
        if total_train_step % 50 == 0:
             writer.add_scalar('train_loss', loss.item(), total_train_step)
             writer.add_scalar('train_log_rmse', log_rmse(output, y), total_train_step)

    avg_loss = total_loss / num_batches
    avg_rmse = total_rmse / num_batches

    # 保存最佳模型
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        os.makedirs('./models', exist_ok=True)
        torch.save(net.state_dict(), './models/house_price1.pth')

    #每10个epoch打印一次
    if(epoch +1)% 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}")
        writer.add_scalar('epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('epoch_log_rmse', avg_rmse, epoch+1)

print(f"\n训练完成!最佳 Log RMSE: {best_rmse:.4f}")
print("最佳模型已保存为 house_price1.pth")

## =========== 开始测试 ==========
print("\n"+"="*50)
print("开始测试")

#加载模型
net.load_state_dict(torch.load('./models/house_price1.pth'))
net.to(device)
print(" ✅ 模型加载成功！")

# 预测
#all_preds存放所有测试集样本的预测房价
all_preds = []
net.eval()
with torch.no_grad():
    for X in test_loader:
        #test_loader 返回的是 (数据,) 的元组 无标签所以需要X[0]返回数据
        x = X[0].to(device)
        outputs = net(x)
        all_preds.append(outputs.cpu().numpy())

    #print(all_preds)

# 合并所有预测结果: all_preds是全步的房价按批次排列,合并后为一组数据
# 按行拼接所有array,然后展品
preds = np.concatenate(all_preds,axis=0).flatten()
print(f"✅ 预测完成！共 {len(preds)} 个样本")

# 生成 Kaggle 提交文件
submission = pd.DataFrame({
    'Id': test_data['Id'],  # 测试集的 Id
    'SalePrice': preds      # 预测的房价
})

# 确保 SalePrice 都是正数
submission['SalePrice'] = submission['SalePrice'].clip(lower=0)

# 保存为 CSV
os.makedirs('./submissions', exist_ok=True)
submission.to_csv('./submissions/submission1.csv', index=False)
print(f"\n✅ 提交文件已保存: ./submissions/submission1.csv")

# 查看前 10 个预测结果

writer.close()
print("\n🎉 全部完成！")
print("在终端运行: tensorboard --logdir=runs")
print("然后去 Kaggle 提交 submission.csv")












