import torch
import torch.nn as nn
import warnings

from ultralytics.nn.modules.block import RepBottleneck

warnings.filterwarnings('ignore')
from ultralytics.nn.modules import C3
__all__ = ['C2f_DCNv4','C3k2_DCNv4','DCNV4']
'''
CVPR 2024 
本文介绍了可变形卷积v4（DCNv4），这是一种为广泛的视觉应用而设计的高效和有效的卷积算子。
DCNv4解决了其前身DCNv3的局限性，有两个关键的增强： 
1.在空间聚合中的softmax规范化以增强其动态特性和表达能力；
2.优化内存访问以减少冗余操作以加速访问速度。与DCNv3相比，这些改进使得收敛速度显著提高，DCNv4实现了前向传播速度的三倍以上。

DCNv4在各种任务中都表现出了卓越的性能，包括图像分类、实例和语义分割等所有cv任务，特别是图像生成。
当集成到潜在扩散模型中的U-Net等生成模型中时，DCNv4的性能优于其基线，强调了其增强生成模型的可能性。

DCNv4模块的原理及作用
原理：DCNv4引入了一种动态、稀疏的卷积操作，允许卷积核根据输入特征动态采样，并在自适应的窗口内聚合空间信息。
这一过程不再需要softmax归一化，因此聚合权重具有更大的动态范围，增强了表达能力。
此外，DCNv4通过减少内存访问成本和消除多余计算，大幅提高了运行效率。

作用：DCNv4被广泛应用于图像分类、实例分割、语义分割等任务中，并且在生成模型（如U-Net的潜在扩散模型）中表现出色。
与其前代版本相比，DCNv4不仅显著加快了收敛速度，还提高了推理效率，成为了现代视觉模型中一种高效的基础模块

'''
try:
    from DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass
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
class DCNV4(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, act=True):
        super().__init__()
        self.inc = inc
        self.ouc = ouc
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if self.inc !=self.ouc:
            x = self.stem_conv(x)
        x = self.dcnv4(x,(x.size(2), x.size(3)))
        x = self.act(self.bn(x))
        return x
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
class Bottleneck_DCNV4(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV4(c_, c2, k[1])
        # self.cv2 = DCNv4(c2, k[1])
class C2f_DCNv4(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV4(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(Bottleneck_DCNV4(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
class C3k2_DCNv4(C2f_DCNv4):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DCNV4(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
try:
    from DCNv4.modules.dcnv4 import DCNv4
except ImportError as e:
    pass
if __name__ == "__main__":
    input = torch.randn(1, 64, 32, 32).cuda()
    # model = C3k2_DCNv4(64,128).cuda()
    model =  DCNv4(64,3,2).cuda()
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

