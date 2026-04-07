import torch
import torch.nn as nn

class ElementScale(nn.Module):
    def __init__(self, channels, init_value=1e-5):
        super().__init__()
        # 用于缩放通道的可学习参数
        self.scale = nn.Parameter(torch.ones([channels, 1, 1]) * init_value)

    def forward(self, x):
        return x * self.scale


class CA_Block(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        # 默认输出通道数等于输入通道数
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        # 第一层卷积：1x1卷积调整通道数
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        
        # Depthwise Convolution (DWConv)，逐通道卷积
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, 
                                 bias=True, groups=hidden_features)
        
        # 激活函数
        self.act = act_layer()
        
        # 第二层卷积：1x1卷积将通道数恢复
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        
        # Dropout层
        self.drop = nn.Dropout(drop)

        # 通道分解模块：用于计算通道的注意力
        self.decompose = nn.Conv2d(hidden_features, 1, kernel_size=1)

        # 可学习的缩放因子：用于通道注意力
        self.sigma = ElementScale(hidden_features, init_value=1e-5)

    def forward(self, x):
        # 第一层卷积，分割通道
        x, v = self.fc1(x).chunk(2, dim=1)

        # 逐通道卷积和激活
        x = self.act(self.dwconv(x) + x)

        # 通道注意力机制：通过与v通道的加权相乘
        x = self.act(x) * v
        
        # Dropout
        x = self.drop(x)

        # 通道分解（用于通道注意力机制）
        decompose = self.decompose(x)
        
        # 通道注意力机制：结合decompose，进行缩放操作
        x = x + self.sigma(x - decompose)

        # 第二层卷积，恢复输出通道数
        x = self.fc2(x)

        # Dropout
        x = self.drop(x)

        return x


