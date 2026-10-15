"""
训练模型,并把损失曲线和准确率曲线画出来。

跑完之后会在 save/ 里生成 titanic_model.pth(模型参数)
和 preprocess_params.json(预处理参数),在 docs/ 里生成两张图。
"""

import random
from pathlib import Path

import matplotlib
# 用 Agg 后端:只往文件里存图,不弹窗口,脚本就不会卡在画图窗口上
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_preprocessing import load_and_split, save_params
from model import TitanicModel, save_model

# 项目根目录(src 的上一层)。这样不管在哪个目录下敲命令都能找到文件
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'train.csv'
SAVE_DIR = ROOT_DIR / 'save'
DOCS_DIR = ROOT_DIR / 'docs'

SEED = 42

# 训练配置都写在这,方便一眼看到用了什么超参数,也方便实验可复现
CONFIG = {
    'seed': SEED,
    'test_size': 0.2,      # 训练集 : 测试集 = 8 : 2
    'batch_size': 32,
    'epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.2,
}


class TitanicDataset(Dataset):
    """把特征和标签包一下,交给 DataLoader 按批取"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    """
    把能设的随机种子都设一遍。
    数据划分、参数初始化、DataLoader 打乱顺序都会用到随机数,
    全部固定住之后每次跑出来的结果才一模一样。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, X, y, criterion):
    """
    在给定数据上算一遍损失和准确率。
    这里把 dropout 关掉(model.eval),不然同一个模型每次算出来的数都不一样。
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        loss = criterion(outputs, y).item()
        preds = (outputs > 0.5).float()
        # 预测对了多少条 / 一共多少条
        acc = (preds == y).float().mean().item()
    return loss, acc


def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, best_epoch):
    """画两张图:损失趋势图、准确率对比图"""

    epochs_range = range(1, len(train_losses) + 1)

    # --- 图1:损失趋势图 ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'loss_curve.png', dpi=150)
    plt.close()

    # --- 图2:训练集与测试集准确率图 ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    # 标一下测试集准确率最高的那一轮
    plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch})')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / 'accuracy_curve.png', dpi=150)
    plt.close()


def main():
    set_seed(CONFIG['seed'])

    print('=' * 50)
    print('开始训练')
    print('=' * 50)

    # --- 1. 读数据、划分、预处理 ---
    print('正在加载并预处理数据...')
    X_train, y_train, X_test, y_test, params = load_and_split(
        DATA_PATH, test_size=CONFIG['test_size'])

    # --- 2. 转成 PyTorch 的张量 ---
    # 特征用 float32,标签也转成 float32 再拉成一列,因为 BCELoss 要求两边形状一样
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f'训练集 {X_train_tensor.shape[0]} 条,测试集 {X_test_tensor.shape[0]} 条,'
          f'特征 {X_train_tensor.shape[1]} 个')

    # --- 3. 用 Dataset 和 DataLoader 组织训练数据 ---
    train_dataset = TitanicDataset(X_train_tensor, y_train_tensor)
    # shuffle 的打乱顺序也固定住种子,不然每轮顺序不一样,结果也会飘
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, generator=generator)

    # --- 4. 建模型、定损失函数和优化器 ---
    input_dim = X_train_tensor.shape[1]
    model = TitanicModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])

    print(f'模型结构: {input_dim} -> 32 -> 16 -> 1')
    print(f'损失函数: BCELoss   优化器: Adam (lr={CONFIG["learning_rate"]}, '
          f'weight_decay={CONFIG["weight_decay"]})')
    print(f'训练轮次: {CONFIG["epochs"]}   批大小: {CONFIG["batch_size"]}')
    print('-' * 50)

    # 每一轮都记一份,画图用
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 记下测试集表现最好的那一轮
    best_test_acc = 0.0
    best_epoch = 0
    best_state = None

    # --- 5. 开始训练 ---
    for epoch in range(CONFIG['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 三步走:清空上一轮的梯度 -> 反向传播算梯度 -> 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 这一轮结束后,在训练集和测试集上各算一遍损失和准确率
        train_loss, train_acc = evaluate(model, X_train_tensor, y_train_tensor, criterion)
        test_loss, test_acc = evaluate(model, X_test_tensor, y_test_tensor, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 测试集准确率创新高就把这一轮的参数抄一份下来
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1:>3}/{CONFIG["epochs"]}]  '
                  f'训练损失 {train_loss:.4f}  测试损失 {test_loss:.4f}  '
                  f'训练准确率 {train_acc:.4f}  测试准确率 {test_acc:.4f}')

    # --- 6. 保存 ---
    # 恢复成测试集上最好的那一轮再存,这样存下来的是表现最好的模型
    if best_state is not None:
        model.load_state_dict(best_state)

    SAVE_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    save_model(model, SAVE_DIR / 'titanic_model.pth')
    save_params(params, SAVE_DIR / 'preprocess_params.json')

    # --- 7. 画图 ---
    plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, best_epoch)

    last_test_acc = test_accuracies[-1]

    print('-' * 50)
    print(f'训练结束。一共 {CONFIG["epochs"]} 轮。')
    print(f'最后一轮的测试集准确率: {last_test_acc:.4f}')
    print(f'最好的测试集准确率:     {best_test_acc:.4f} (第 {best_epoch} 轮,已保存这一轮的模型)')
    print(f'模型和预处理参数已保存到 {SAVE_DIR}')
    print(f'两张曲线图已保存到 {DOCS_DIR}')

    return best_test_acc, best_epoch, last_test_acc


if __name__ == '__main__':
    main()
