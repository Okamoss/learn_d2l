import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# ========== 设备配置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# ========== 2. 定义增广 ==========
train_augs= transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    # transforms.RandomVerticalFlip(), #随机垂直`翻转
    # transforms.RandomRotation(10), # 随机角度变化
    transforms.ToTensor(),
])

# 原始图（无增广）
original_augs = transforms.Compose([
    transforms.ToTensor(),
])


test_augs = transforms.Compose([
    transforms.ToTensor(),
])

# ========== 加载数据 ==========
batch_size = 128

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_augs)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_augs)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 为了实现可视化加载的数据
train_dataset_aug = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_augs)
train_dataset_orig = datasets.CIFAR10(root='./data', train=True, download=True, transform=original_augs)

# ========== 可视化对比 ==========
def show_augmentations():
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))
    fig.suptitle('CIFAR-10 原始图片 vs 水平翻转增广', fontsize=16)

    # 显示6张不同的图片
    for i in range(6):
        # 原始图片（无增广）
        orig_img, label = train_dataset_orig[i]
        # 增广图片（水平翻转）
        aug_img, _ = train_dataset_aug[i]

        # 第一行：原始图片
        axes[0, i].imshow(orig_img.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'原始 {i+1}', fontsize=12)
        axes[0, i].axis('off')

        # 第二行：增广图片
        axes[1, i].imshow(aug_img.permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'翻转 {i+1}', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('./cifar10_flip_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("对比图已保存到 ./cifar10_flip_comparison.png")


#show_augmentations()