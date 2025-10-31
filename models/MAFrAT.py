import torch
import torch.nn.functional as F
from torch import nn
from .FrFT import frft, ifrft


# ----------------------
# 基础卷积构件
# ----------------------
class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = [
            Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1),
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_shortcut else self.conv(x)


class InvertedDepthWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        layers = [
            Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1),
            Conv1dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_shortcut else self.conv(x)


# ----------------------
# PNet：自适应分数阶估计网络
# ----------------------
class PNet(nn.Module):
    """
    Adaptive Fractional Estimation Network
    输入特征 → 两次下采样 → 两个残差块 → 全局池化 → 输出分数阶 p
    """
    def __init__(self, in_ch, base_ch):
        super().__init__()

        # 下采样阶段
        self.conv1 = Conv2dGNGELU(in_ch, base_ch, kernel_size=3, stride=2)
        self.conv2 = InvertedDepthWiseConv2d(base_ch, base_ch * 2, kernel_size=3, stride=2)

        # 残差块
        self.res = nn.Sequential(
            InvertedDepthWiseConv2d(base_ch * 2, base_ch * 2),
            InvertedDepthWiseConv2d(base_ch * 2, base_ch * 2),
        )

        # 全局回归头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 2, base_ch, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(base_ch, 1, 1, bias=True),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        # 下采样特征提取
        x = self.conv1(x)
        x = self.conv2(x)

        # 残差增强
        res = self.res(x)
        x = x + res

        # 全局回归得到 p
        p = self.head(x)  # (B,1,1,1)
        return p.view(x.size(0))  # 输出形状 (B,)


# ----------------------
# MAFrAT：Multi-Axis Fractional Fourier Attention
# ----------------------
class MAFrAT(nn.Module):
    def __init__(self, dim, bias=False, a=16, b=16, c_h=16, c_w=16):
        super().__init__()
        self.dim = dim
        self.a_weight = nn.Parameter(torch.ones(2, 1, dim // 4, a))
        self.b_weight = nn.Parameter(torch.ones(2, 1, dim // 4, b))
        self.c_weight = nn.Parameter(torch.ones(2, dim // 4, c_h, c_w))
        self.dw_conv = InvertedDepthWiseConv2d(dim // 4, dim // 4)
        self.wg_a = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_b = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )
        self.wg_c = nn.Sequential(
            InvertedDepthWiseConv2d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, dim // 4),
        )
        self.pnet = PNet(in_ch=dim, base_ch= dim * 2)

    def forward(self, x):
        # --- 支持两种输入格式 ---
        if x.dim() == 3:  # (B, N, C)
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot reshape sequence length {N} to (H,W)"
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            seq_mode = True
        else:
            seq_mode = False
            B, C, H, W = x.shape

        # --- 自适应分数阶 ---
        p = self.pnet(x).mean().detach()


        # --- 四分支分块 ---
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, c, a, b = x1.size()

        # --- 分支 a ---
        x1 = x1.permute(0, 2, 1, 3)  # B,a,c,b
        x1 = frft(x1, p)
        a_weight = self.wg_a(F.interpolate(self.a_weight, size=x1.shape[2:4],mode='bilinear', align_corners=True).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)

        a_weight = torch.view_as_complex(a_weight.contiguous())
        x1 = x1 * a_weight
        x1 = ifrft(x1, p).permute(0, 2, 1, 3)
        x1 = torch.abs(x1)

        # --- 分支 b ---
        x2 = x2.permute(0, 3, 1, 2)  # B,b,c,a
        x2 = frft(x2, p)
        b_weight = self.wg_b(F.interpolate(self.b_weight, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        b_weight = torch.view_as_complex(b_weight.contiguous())
        x2 = x2 * b_weight
        x2 = ifrft(x2, p).permute(0, 2, 1, 3)
        x2 = torch.abs(x2)

        # --- 分支 c ---
        x3 = frft(x3, p)
        c_weight = self.wg_c(F.interpolate(self.c_weight, size=x3.shape[2:4],mode='bilinear', align_corners=True)).permute(1, 2, 3, 0)
        c_weight = torch.view_as_complex(c_weight.contiguous())
        x3 = x3 * c_weight
        x3 = ifrft(x3, p)
        x3 = torch.abs(x3)

        # --- 分支 d ---
        x4 = self.dw_conv(x4)

        # --- 融合 ---
        out = torch.cat([x1, x2, x3, x4], dim=1)

        # --- 若输入是序列，恢复为 (B, N, C) ---
        if seq_mode:
            out = out.flatten(2).transpose(1, 2)
        return out
