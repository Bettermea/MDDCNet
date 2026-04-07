import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class MultiScalePixelAdaptiveDilatedConv(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        # 多尺度 offset 预测
        self.offset_conv = nn.Conv2d(
            channels, 
            2 * kernel_size * kernel_size * len(dilations), 
            kernel_size=3, 
            padding=1
        )

        # 为每个 dilation 建立 deformable 分支
        self.deform_convs = nn.ModuleList([
            DeformConv2d(
                channels, channels, 
                kernel_size=kernel_size, 
                padding=d * (kernel_size // 2), 
                dilation=d, bias=False
            ) for d in dilations
        ])

        # 融合卷积
        self.fuse = nn.Conv2d(len(dilations) * channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset_conv(x)  # (B, 2*Kh*Kw*len(dilations), H, W)

        # 将 offset 拆成多尺度分支
        offsets = torch.chunk(offset, len(self.dilations), dim=1)

        out_branches = []
        for i, d in enumerate(self.dilations):
            out = self.deform_convs[i](x, offsets[i])
            out_branches.append(out)

        # 融合多尺度输出
        out = torch.cat(out_branches, dim=1)
        out = self.fuse(out)
        return out



class MSGDC(nn.Module):
    def __init__(self, dim, reduction=16, dilations=(1, 2, 3)):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

        # SE 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # 多尺度像素级自适应卷积
        self.adapt_conv = MultiScalePixelAdaptiveDilatedConv(dim, dilations=dilations)

        # FFN 分支
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.SiLU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        identity = x
        out = self.act(self.norm(x))

        # 残差缩放
        out = identity + self.res_scale * out

        # 多尺度自适应卷积
        out = out + self.adapt_conv(out)

        # 通道注意力
        out = out * self.se(out)

        # FFN
        out = out + self.ffn(out)

        return out


# ------------------------------------------------
# 测试代码
# ------------------------------------------------
if __name__ == "__main__":
    x = torch.randn(1, 128, 80, 80)
    block = MSGDC(128)
    y = block(x)
    print(f"Input: {x.shape} → Output: {y.shape}")

