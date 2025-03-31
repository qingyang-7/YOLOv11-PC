import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from mmcv.cnn import build_norm_layer
act_layer = nn.ReLU
ls_init_value = 1e-6
__all__=['C2f_CED','C2f_LRCED']
#https://github.com/zhanggang001/CEDNet/blob/cednet/models/cednet.py
#https://arxiv.org/abs/2302.06052
'''
CEDNet：用于密集预测的级联编码器-解码器网络
即插即用多尺度特征提取模块：CED 和 LRCED     

多尺度特征对于密集的预测任务（例如对象检测、实例分割和语义分割）至关重要。
流行的方法通常利用分类主干来提取多尺度特征，然后使用轻量级模块（例如，FPN 和 BiFPN 中的融合模块，
两种典型的目标检测方法）融合这些特征。然而，由于这些方法将大部分计算资源分配给了分类主干，
因此这些方法中的多尺度特征融合被延迟，这可能导致特征融合不充分。虽然有些方法从早期阶段就执行特征融合，
但它们要么无法充分利用高级特征来指导低级特征学习，要么结构复杂，导致性能欠佳。

我们提出了一种简化的级联编码器-解码器网络，称为 CEDNet。CEDNet 中的所有阶段共享相同的编码器-解码器结构，
并在解码器内执行多尺度特征融合。CEDNet 的一个特点是它能够从早期阶段就整合高级特征，以指导后续阶段的低级特征学习
，从而提高多尺度特征融合的有效性。通过广泛实验证明了我们方法对目标检测、实例分割和语义分割的有效性。

CED模块的作用： 
CED模块在该网络结构中作为基本的构建单元，主要用于多尺度特征提取。
每个CED块由以下两个主要部分组成：
Token Mixer：它通过轻量级的7x7深度卷积执行空间特征的交互。通过这种方式，CED块能够在空间维度上进行高效的特征提取。
MLP：MLP包含两个全连接层，用于在通道维度上进行特征交互，帮助网络在不同通道之间提取丰富的特征表示。
通过这些操作，CED模块在不同的尺度上提取特征，并在网络的多个阶段中进行逐步特征融合，以实现多尺度的有效信息提取。

LRCED模块的作用：
LRCED块是在CED模块基础上的改进。其核心特点是增加了长距离依赖的提取能力。
通过引入7x7扩张深度卷积，该模块可以捕获长距离的空间特征。
这使得网络在低分辨率的特征层级仍然能够保留关键的空间信息，并增强对全局上下文的理解。
LRCED模块通过增加感受野，帮助模型在提取粗略特征时能够捕获更多的上下文信息。
这对密集预测任务尤为重要，例如在图像分割或目标检测任务中，它有助于模型在低分辨率层级保持丰富的空间特征信息。

'''


class CED(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=dict(type='BN') , **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)    # input (N, C, *)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LRCED(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg= dict(type='BN') , **kwargs):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
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

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C2f(nn.Module):
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
class C2f_CED(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CED(self.c) for _ in range(n))

class C2f_LRCED(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(LRCED(self.c) for _ in range(n))
if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32)
    model = CED(64)
    # model = LRCED(64)
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
