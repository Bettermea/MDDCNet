import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
__all__ = ['BiFPN_Concat']
 
 
# def autopad(k, p=None, d=1):
#     """
#     Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
#     `k`: kernel, `p`: padding, `d`: dilation.
#     """
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
 
 
# class Conv(nn.Module):
#     # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
#     default_act = nn.SiLU()  # default activation
 
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initializes a standard convolution layer with optional batch normalization and activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
#     def forward(self, x):
#         """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
#         return self.act(self.bn(self.conv(x)))
 
#     def forward_fuse(self, x):
#         """Applies a fused convolution and activation function to the input tensor `x`."""
#         return self.act(self.conv(x))

# class MLCA(nn.Module):
#     def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
#         super(MLCA, self).__init__()
 
#         # ECA 计算方法
#         self.local_size = local_size
#         self.gamma = gamma
#         self.b = b
#         t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
#         k = t if t % 2 else t + 1
 
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
#         self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
 
#         self.local_weight = local_weight
 
#         self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
#         self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
 
#     def forward(self, x):
#         local_arv = self.local_arv_pool(x)
#         global_arv = self.global_arv_pool(local_arv)
 
#         b, c, m, n = x.shape
#         b_local, c_local, m_local, n_local = local_arv.shape
 
#         # (b,c,local_size,local_size) -> (b,c,local_size*local_size) -> (b,local_size*local_size,c) -> (b,1,local_size*local_size*c)
#         temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
#         # (b,c,1,1) -> (b,c,1) -> (b,1,c)
#         temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
 
#         y_local = self.conv_local(temp_local)
#         y_global = self.conv(temp_global)
 
#         # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
#         y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c,
#                                                                                                             self.local_size,
#                                                                                                             self.local_size)
#         # (b,1,c) -> (b,c,1) -> (b,c,1,1)
#         y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)
 
#         # 反池化
#         att_local = y_local_transpose.sigmoid()
#         att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
#         att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
 
#         x = x * att_all
#         return x



# class BiFPN_Concat(nn.Module):
#     def __init__(self, dimension=1):
#         super(BiFPN_Concat, self).__init__()
#         self.d = dimension
#         self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#         self.MLCA = None  # 初始化时暂不定义 MLCA
#         self._mlca_initialized = False  # 标志变量，用于检查 MLCA 是否已初始化
 
#     def forward(self, x):
#         w = self.w
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
#         # Fast normalized fusion
#         x = [weight[0] * x[0], weight[1] * x[1]]
#         x=torch.cat(x,self.d)
        
#         # 如果 MLCA 尚未初始化，则根据输入特征图的通道数动态初始化
#         if not self._mlca_initialized:
#             in_size = x.size(1)  # 获取拼接后的通道数
#             self.mlca = MLCA(in_size).to('cuda')  # 初始化 MLCA
#             self._mlca_initialized = True  # 设置标志变量为 True，表示 MLCA 已初始化

#         x = self.mlca(x)  # 调用 MLCA 模块
#         return x
    

import math



class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()
 
        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1
 
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
 
        self.local_weight = local_weight
 
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)
 
    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)
 
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
 
        # (b,c,local_size,local_size) -> (b,c,local_size*local_size) -> (b,local_size*local_size,c) -> (b,1,local_size*local_size*c)
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        # (b,c,1,1) -> (b,c,1) -> (b,1,c)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
 
        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)
 
        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c,
                                                                                                            self.local_size,
                                                                                                            self.local_size)
        # (b,1,c) -> (b,c,1) -> (b,c,1,1)
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)
 
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
 
        x = x * att_all
        return x

class ChannelPool(nn.Module):
    """通道池化模块：拼接全局最大池化与全局平均池化结果，增强特征表达
    作用：融合两种池化方式的优势，为空间注意力提供更全面的特征统计信息
    """
    def forward(self, x):
        # 对输入特征沿通道维度（dim=1）分别执行最大池化和平均池化，再扩展通道维度（unsqueeze(1)）后拼接
        # 输出形状：(B, 2, H, W)，2个通道分别对应max_pool和avg_pool结果
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),  # 全局最大池化（保留通道维度最大值）
             torch.mean(x, 1).unsqueeze(1)),   # 全局平均池化（保留通道维度平均值）
            dim=1  # 沿通道维度拼接
        )


class Basic(nn.Module):
    """基础卷积模块：封装"卷积+批归一化+激活函数"的通用结构
    作用：提供标准化的特征变换单元，可灵活配置是否使用BN和激活函数
    
    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 卷积步长，默认1
        padding (int): 卷积填充，默认0
        relu (bool): 是否使用LeakyReLU激活，默认True
        bn (bool): 是否使用批归一化，默认True
        bias (bool): 卷积层是否使用偏置，默认False（BN层已包含偏置效果）
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes  # 记录输出通道数，便于后续模块适配
        
        # 核心卷积层
        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # 批归一化层（可选）
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        
        # 激活函数层（可选，默认LeakyReLU，缓解梯度消失）
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)  # 卷积变换
        if self.bn is not None:
            x = self.bn(x)  # 批归一化（稳定训练）
        if self.relu is not None:
            x = self.relu(x)  # 激活函数（引入非线性）
        return x


class CALayer(nn.Module):
    """通道注意力层（CA）：结合行列维度池化与激发操作，增强通道相关性
    创新点：不仅关注通道全局信息，还分别对行（H）和列（W）维度建模，提升通道注意力的精细度
    
    Args:
        channel (int): 输入通道数
        reduction (int): 通道压缩系数，默认16（平衡计算量与表达能力）
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        
        # 行维度池化：沿高度（H）池化，保留宽度（W）信息（输出形状：B, C, 1, W）
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))
        
        # 1×1卷积：融合池化后的行维度特征（输入2C通道：avg+max）
        self.conv_1x1 = nn.Conv2d(
            in_channels=2 * channel,
            out_channels=2 * channel,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)  # 批归一化
        self.Relu = nn.LeakyReLU()  # 激活函数
        
        # 行维度激发操作（F_w）：压缩通道后恢复，生成行维度注意力权重
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 通道压缩
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 通道恢复
        )
        
        # 列维度激发操作（F_h）：与F_w结构一致，生成列维度注意力权重
        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        
        self.sigmoid = nn.Sigmoid()  # 归一化注意力权重到[0,1]

    def forward(self, x):
        N, C, H, W = x.size()  # N=批量，C=通道，H=高度，W=宽度
        res = x  # 保存残差连接
        
        # 步骤1：行维度池化与特征融合
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)  # 拼接avg/max池化结果（2C通道）
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))  # 1×1卷积+BN+激活，融合特征
        
        # 步骤2：拆分行列特征并生成注意力权重
        x_h, x_w = x.split(C, 1)  # 按通道拆分为两部分（各C通道，对应行/列）
        x_h = self.F_h(x_h)  # 列维度激发，生成列注意力权重（B, C, 1, W）
        x_w = self.F_w(x_w)  # 行维度激发，生成行注意力权重（B, C, 1, W）
        
        # 步骤3：注意力加权与残差连接
        s_h = self.sigmoid(x_h).expand_as(res)  # 扩展列权重到输入尺寸
        s_w = self.sigmoid(x_w).expand_as(res)  # 扩展行权重到输入尺寸
        out = res * s_h * s_w  # 残差特征分别乘以行列注意力权重
        
        return out


class spatial_attn_layer(nn.Module):
    """空间注意力层（SA）：基于通道池化与卷积生成空间注意力权重
    作用：聚焦图像中的重要空间区域（如目标、边缘），抑制背景冗余信息
    
    Args:
        kernel_size (int): 卷积核大小，默认3（平衡局部与全局空间建模）
    """
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()  # 通道池化（融合max/avg池化）
        
        # 基础卷积：将2通道池化结果映射为1通道空间注意力图
        self.spatial = Basic(
            in_planes=2,
            out_planes=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # SAME padding，保持尺寸
            bn=False,  # 空间注意力无需BN，避免丢失局部细节
            relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)  # 通道池化：(B, C, H, W)→(B, 2, H, W)
        x_out = self.spatial(x_compress)  # 卷积生成空间权重图：(B, 2, H, W)→(B, 1, H, W)
        scale = torch.sigmoid(x_out)  # 归一化权重到[0,1]
        return x * scale  # 空间权重施加到输入特征


class RCSSC(nn.Module):
    """残差通道-空间-尺度组合模块（RCSSC）：融合通道注意力、空间注意力与尺度信息，增强特征表达
    核心逻辑：通过"头部卷积→双注意力分支→尺度融合→特征整合→残差连接"，实现多维度特征优化
    
    Args:
        n_feat (int): 输入/输出特征通道数
        reduction (int): 通道注意力的压缩系数，默认16
    """
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4  # 尺度池化系数（下采样比例）
        
        # 头部卷积：初步特征变换，为后续注意力模块做准备
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),  # 激活函数引入非线性
        )
        
        # 尺度通道模块（SC）：下采样后提取大尺度特征，增强全局语义
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),  # 下采样4倍（H/W→H/4/W/4）
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),  # 卷积提取特征
            nn.BatchNorm2d(n_feat)  # 批归一化稳定训练
        )
        
        self.SA = spatial_attn_layer()  # 空间注意力分支
        # self.CA = CALayer(n_feat, reduction)  # 通道注意力分支
        self.CA=MLCA(n_feat)
        
        # 特征整合模块：拼接双注意力结果，通过1×1+3×3卷积融合
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),  # 1×1卷积压缩通道（2n_feat→n_feat）
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)  # 3×3卷积增强局部关联
        )
        
        self.ReLU = nn.LeakyReLU()  # 最终激活函数
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)  # 尾部卷积优化输出

    def forward(self, x):
        res = x  # 保存残差连接
        
        # 步骤1：头部特征变换
        x = self.head(x)  # (B, n_feat, H, W)→(B, n_feat, H, W)
        
        # 步骤2：双注意力分支计算
        sa_branch = self.SA(x)  # 空间注意力分支输出
        ca_branch = self.CA(x)  # 通道注意力分支输出
        
        # 步骤3：双注意力特征融合
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 沿通道拼接（2n_feat通道）
        x1 = self.conv1x1(x1)  # 卷积融合为n_feat通道
        
        # 步骤4：尺度特征融合
        # SC模块：下采样提取大尺度特征→上采样恢复尺寸→与原始特征相加→sigmoid生成尺度权重
        x2 = torch.sigmoid(torch.add(x, F.interpolate(self.SC(x), x.size()[2:])))
        
        # 步骤5：特征加权与残差连接
        out = torch.mul(x1, x2)  # 注意力融合特征 × 尺度权重
        out = self.tail(out)  # 尾部卷积优化
        out = out + res  # 残差连接（稳定训练，保留原始特征）
        out = self.ReLU(out)  # 激活函数增强表达
        
        return out

class RCSSC_Concat(nn.Module):
    def __init__(self, dimension=1):
        super(RCSSC_Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # x 是一个 list，例如 [x1, x2, x3]
        ref = x[0]
        resized = [F.interpolate(i, size=ref.shape[2:], mode='nearest') for i in x]
        return torch.cat(resized, dim=self.d)




if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模拟两个分支输入
    x1 = torch.randn(1, 64, 64, 64).to(device)
    x2 = torch.randn(1, 64, 64, 64).to(device)

    # 初始化模块（两个分支每个都有 64 个通道）
    rcssc_concat = RCSSC_Concat(in_channels=64).to(device)

    # 前向传播
    y = rcssc_concat([x1, x2])

    print("输入分支维度：", x1.shape)
    print("输出拼接维度：", y.shape)



