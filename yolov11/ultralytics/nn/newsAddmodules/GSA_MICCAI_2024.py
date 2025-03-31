import torch
import torch.nn as nn
# 代码：https://github.com/xuxuxuxuxuxjh/LB-UNet/blob/main/lbunet.py
# 论文：https://papers.miccai.org/miccai-2024/paper/2135_paper.pdf
__all__= ['Group_shuffle_Attention','C2f_GSA','C3k2_GSA']
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math

from ultralytics.nn.modules.block import RepBottleneck

from ultralytics.nn.modules import C3


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Group_shuffle_Attention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        c_dim = dim_in // 4

        self.share_space1 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space2 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space3 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )
        self.share_space4 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=3, padding=1, groups=c_dim),
            nn.GELU(),
            nn.Conv2d(c_dim, c_dim, 1)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        x1 = x1 * self.conv1(F.interpolate(self.share_space1, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x2 = x2 * self.conv2(F.interpolate(self.share_space2, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x3 = x3 * self.conv3(F.interpolate(self.share_space3, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x4 = x4 * self.conv4(F.interpolate(self.share_space4, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x = torch.cat([x2, x4, x1, x3], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = Group_shuffle_Attention(c2,c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_GSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_GSA(C2f_GSA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64) #随机生成一张输入图片张量
    # 初始化GSA模块并设定通道维度
    gsa  = Group_shuffle_Attention(32,64)
    output = gsa(input)  # 进行前向传播
    # 输出结果的形状
    print("GSA_输入张量的形状：", input.shape)
    print("GSA_输出张量的形状：", output.shape)