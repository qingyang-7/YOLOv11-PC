import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3

__all__=['PSConv','C2f_PSConv','C3k2_PSConv']
# 论文地址：https://arxiv.org/pdf/2412.16986

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
        self.cv2 = Conv(c_, c2, k[1], 1, g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class PSConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # 定义4种非对称填充方式，用于风车形状卷积的实现
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # 每个元组表示 (左, 上, 右, 下) 填充
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # 创建4个填充层

        # 定义水平方向卷积操作，卷积核大小为 (1, k)，步幅为 s，输出通道数为 c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # 定义垂直方向卷积操作，卷积核大小为 (k, 1)，步幅为 s，输出通道数为 c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # 最终合并卷积结果的卷积层，卷积核大小为 (2, 2)，输出通道数为 c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # 对输入 x 进行不同填充和卷积操作，得到四个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 水平方向，第一个填充方式
        yw1 = self.cw(self.pad[1](x))  # 水平方向，第二个填充方式
        yh0 = self.ch(self.pad[2](x))  # 垂直方向，第一个填充方式
        yh1 = self.ch(self.pad[3](x))  # 垂直方向，第二个填充方式

        # 将四个卷积结果在通道维度拼接，并通过一个额外的卷积层处理，最终输出
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # 在通道维度拼接，并通过 cat 卷积层处理
class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()

        self.psconv = PSConv(c1,c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""

        return x + self.psconv(x) if self.add else self.psconv(x)
class C2f_PSConv(nn.Module):
    """Faster Implementation of APCSP Bottleneck with Asymmetric Padding convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, P=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if P:
            self.m = nn.ModuleList(
                APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
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
        self.m = nn.Sequential(*(APBottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
class C3k2_PSConv(C2f_PSConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    module =  PSConv(c1=64,c2=128,k=3,s=2)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
