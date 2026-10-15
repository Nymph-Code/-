"""
二分类模型的结构。

输入:10 个特征
输出:1 个数,经过 Sigmoid 之后是 0~1 之间的生存概率,大于 0.5 就算幸存
"""

import torch
import torch.nn as nn


class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()

        # 输入层(10) -> 隐藏层1(32) -> 隐藏层2(16) -> 输出层(1)
        # 一开始我只用了一层 16 个神经元,感觉有点不够用,就加宽了一点,
        # 两层之间加了 Dropout 让一部分神经元随机失效,压一压过拟合。
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def save_model(model, path):
    """只存网络的参数,结构在代码里已经写死了"""
    torch.save(model.state_dict(), path)


def load_model(path, input_dim):
    """
    按和训练时一样的结构先建一个空模型,再把存好的参数读进去。
    这样预测的时候就不用重新写一遍网络结构了。
    """
    model = TitanicModel(input_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
