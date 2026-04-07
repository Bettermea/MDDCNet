from .common_utils_mbyolo import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath

__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock")



class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,

    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4
        self.scan_stride = 3 # ES2D间隔采样步长
        self.scan_tokens = 0  # 输出的scan_tokens数目
        self.token_printed = False

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)

        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    # 原forword
    # def forward(self, x: torch.Tensor, **kwargs):
    #     x = self.in_proj(x)
    #     if not self.disable_z:
    #         x, z = x.chunk(2, dim=1)  # (b, d, h, w)

    #         if not self.disable_z_act:
    #             z1 = self.act(z)
    #     if self.d_conv > 0:
    #         x = self.conv2d(x)  # (b, d, h, w)
    #     x = self.act(x)
    #     y = self.forward_core(x, channel_first=(self.d_conv > 1))
    #     y = y.permute(0, 3, 1, 2).contiguous()
    #     if not self.disable_z:
    #         y = y * z1
    #     out = self.dropout(self.out_proj(y))
    #     return out


    # def forward(self, x: torch.Tensor, **kwargs):

    #     stride = 2  # ES2D stride

    #     x = self.in_proj(x)

    #     if not self.disable_z:
    #         x, z = x.chunk(2, dim=1)  # (b, d, h, w)

    #         # 记录原始尺寸
    #         H, W = x.shape[2], x.shape[3]

    #         # ES2D 空洞采样 (下采样)
    #         x = x[:, :, ::stride, ::stride]
    #         z = z[:, :, ::stride, ::stride]

    #         if not self.disable_z_act:
    #             z1 = self.act(z)

    #     if self.d_conv > 0:
    #         x = self.conv2d(x)

    #     x = self.act(x)

    #     # SSM 扫描
    #     y = self.forward_core(x, channel_first=(self.d_conv > 1))
    #     y = y.permute(0, 3, 1, 2).contiguous()

    #     # 恢复尺寸 (上采样)
    #     if stride > 1:
    #         y = F.interpolate(y, size=(H, W), mode="nearest")
    #         z1 = F.interpolate(z1, size=(H, W), mode="nearest")

    #     if not self.disable_z:
    #         y = y * z1

    #     out = self.dropout(self.out_proj(y))

    #     return out

    def forward(self, x: torch.Tensor, **kwargs):

        stride = self.scan_stride

        x = self.in_proj(x)

        if not self.disable_z:
            x, z = x.chunk(2, dim=1)

            if not self.disable_z_act:
                z1 = self.act(z)
            else:
                z1 = z

        # depthwise conv
        if self.d_conv > 1:
            x = self.conv2d(x)

        x = self.act(x)

        B, C, H, W = x.shape

        # ===== DSS2D sparse sampling =====
        if stride > 1:
            xs = x[:, :, ::stride, ::stride]
        else:
            xs = x
        self.scan_tokens += xs.shape[2] * xs.shape[3] #统计scan_tokens数目

        # SSM scan
        y = self.forward_core(xs, channel_first=True)

        y = y.permute(0, 3, 1, 2).contiguous()

        # ===== restore resolution =====
        if stride > 1:
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        if not self.disable_z:
            y = y * z1

        out = self.dropout(self.out_proj(y))

        if not self.token_printed:
            print("scan tokens总数目", self.scan_tokens)
            self.token_printed = True

        return out


class ElementScale(nn.Module):
    def __init__(self, channels, init_value=1e-5):
        super().__init__()
        # 用于缩放通道的可学习参数
        self.scale = nn.Parameter(torch.ones([channels, 1, 1]) * init_value)
    def forward(self, x):
        return x * self.scale


class Improved_CA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        # 升维
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        
        # Depthwise卷积
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        
        # 原始 MoGA-Net 通道分解
        self.decompose = nn.Conv2d(hidden_features, 1, kernel_size=1)
        self.sigma = ElementScale(hidden_features)
        
        # 新增全局通道注意力分支
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.fc1(x)
        
        # DWConv + 残差
        x_dw = self.dwconv(x)
        x = self.act(x + x_dw)
        
        # 通道注意力融合
        local_attn = self.act(self.decompose(x))
        attn = self.sigma(x - local_attn) * self.global_attn(x)
        x = x * (1 + attn)
        
        # 输出
        x = self.fc2(x)
        x = self.drop(x)
        return x

class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
                 # in_features=输入通道数；hidden_features=中间隐藏层通道数；out_features=输出通道数；
        super().__init__()
        out_features = out_features or in_features #输出通道数=输入通道数
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        
        # 普通FFN结构
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # 去掉*2，不需要拆分成两部分
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 直接全连接，不拆分
        x = self.act(self.dwconv(x) + x)  # 去掉门控乘法，保留残差连接
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InceptionDWConv2d(nn.Module):
    """优化后的深度可分离卷积模块"""
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.0625):
        """
        参数优化说明：
        - branch_ratio: 0.125 → 0.0625 (通道数减半)
        - band_kernel_size: 11 → 7 (减小卷积核)
        - 共享水平/垂直卷积核权重
        """
        super().__init__()
 
        # 计算各分支通道数
        base_gc = int(in_channels * branch_ratio)
        self.split_indexes = (
            in_channels - 3 * base_gc,  # identity分支
            base_gc,  # 方形卷积分支
            base_gc,  # 水平带状分支
            base_gc  # 垂直带状分支
        )
 
        # 分支1：方形卷积
        self.dwconv_hw = nn.Conv2d(
            base_gc, base_gc,
            kernel_size=square_kernel_size,
            padding=square_kernel_size // 2,
            groups=base_gc
        )
 
        # 分支2/3：共享权重的带状卷积
        self.dwconv_shared = nn.Conv2d(
            base_gc, base_gc,
            kernel_size=(band_kernel_size, band_kernel_size),
            padding=band_kernel_size // 2,
            groups=base_gc
        )
 
    def forward(self, x):
        # 分割输入特征
        x_id, x_hw, x_band1, x_band2 = torch.split(x, self.split_indexes, dim=1)
 
        # 各分支处理
        branch_hw = self.dwconv_hw(x_hw)
        branch_band1 = self.dwconv_shared(x_band1)
        branch_band2 = self.dwconv_shared(x_band2)
 
        # 合并结果
        return torch.cat([x_id, branch_hw, branch_band1, branch_band2], dim=1)
 
 
class ConvMlp(nn.Module):
    """优化后的MLP模块"""
 
    def __init__(self, in_features, hidden_ratio=2, act_layer=nn.GELU, drop=0.):
        """
        参数优化：
        - hidden_ratio: 4 → 2 (中间层通道数减半)
        """
        super().__init__()
        hidden_features = int(in_features * hidden_ratio)
 
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_features, in_features, 1)
        )
 
    def forward(self, x):
        return self.net(x)
 
 
class InceptionNeXtBlock(nn.Module):
    """优化后的完整模块"""
    def __init__(self, dim, drop_path=0., mlp_ratio=2):
        """
        参数优化：
        - mlp_ratio: 4 → 2
        """
        super().__init__()
 
        # 深度卷积模块
        self.token_mixer = InceptionDWConv2d(dim)
        self.norm = nn.BatchNorm2d(dim)
 
        # MLP模块
        self.mlp = ConvMlp(dim, hidden_ratio=mlp_ratio)
 
        # 残差连接
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
 
        # 可学习缩放系数
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
 
    def forward(self, x):
        shortcut = x
 
        # 深度卷积分支
        x = self.token_mixer(x)
        x = self.norm(x)
 
        # MLP处理
        x = self.mlp(x)
 
        # 残差连接
        x = x.mul(self.gamma.view(1, -1, 1, 1))
        return self.drop_path(x) + shortcut

class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x





class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)


