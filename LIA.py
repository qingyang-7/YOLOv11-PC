import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3
__all__=['LocalAttention','CLAttention','C3k2_LA','C3k2_CLA','C2f_CLA','C2f_LA']
# 论文链接：https://openaccess.thecvf.com/content/ACCV2024/papers/Wang_PlainUSR_Chasing_Faster_ConvNet_for_Efficient_Super-Resolution_ACCV_2024_paper.pdf
# 代码：https://github.com/icandle/PlainUSR


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool
class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g) #self.gate(x, w)

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
# 二次创新模块 CLAttention 通道局部增强注意力模块  可以冲SCI二区或是三次，B会或是C会
'''
二次创新模块 CLAttention 通道局部增强注意力模块  可以冲SCI二区或是三次，B会或是C会

CLAttention 通道局部增强注意力模块 内容介绍：

首先执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
然后利用LIA局部注意力模块的作用如下：
1.高效的注意力计算：LIA 模块通过计算输入特征的局部重要性（Local Importance），
并结合通道门（Channel Gate）来重新校准注意力图。这种设计实现了2阶信息交互，
但避免了传统2阶注意力（如自注意力）的高计算复杂度。
2.低延迟和性能平衡：LIA 模块通过在下采样的特征图上计算局部重要性，
并利用简单的操作（例如卷积和双线性插值），在提升模型表达能力的同时，显著降低了计算延迟。
3.增强的特征权重化：LIA 使用局部重要性和通道门的组合，
使得模块能够自适应地增强有用的特征，同时弱化无用的特征，从而提高了特征选择的精度。

'''
class CLAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.ca= channel_att(channels)
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        ''' forward '''
        x = self.ca(x)
        # interpolate the heat map
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g  # (w + g) #self.gate(x, w)

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
class Bottleneck_CLA(nn.Module):
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
        self.Attention = CLAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
class Bottleneck_LA(nn.Module):
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
        self.Attention = LocalAttention(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
class C2f_LA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_LA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
class C2f_CLA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_CLA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
class C3k_LA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_LA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
class C3k_CLA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CLA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_LA(C2f_LA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_LA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_LA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
class C3k2_CLA(C2f_CLA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_CLA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CLA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )


if __name__ == "__main__":
    input = torch.randn(1, 30, 128, 128)
    LA = LocalAttention(30)
    output = LA(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

    CLA = CLAttention(30)
    output = CLA(input)
    print('二次创新_CLA_input_size:', input.size())
    print('二次创新_CLA_output_size:', output.size())