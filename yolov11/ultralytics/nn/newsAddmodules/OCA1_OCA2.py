from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn

'''
2023 CCF BigData 中国计算机协会

本文的核心内容是提出了一个名为 OCA（OrthoNet Convolution Attention，正交通道注意力）模块，
旨在通过正交通道的设计和正交化约束，来提升卷积神经网络的特征提取能力和性能表现。
文中还进一步提出了两个正交通道注意力变体模块 OrthoNet block 和 OrthoNet-MOD block，
这两个模块能够有效减少通道之间的冗余性并增加其独立性，从而提升模型的准确性和计算效率。

主要核心内容包括：
正交通道作用：OCA模块基于正交约束，正交化特征可以减少不同特征通道之间的相关性，防止信息冗余并提升模型的表现。
引入注意力机制的作用：在正交通道的基础上，进一步通过注意力机制来增强特征选择的有效性，提升重要特征的权重。
多尺度特征融合：模块能够在不同尺度上处理输入数据，从而有效增强模型对复杂数据、尤其是图像数据的多维度理解能力。
模块的实验验证：本文中通过大量的实验验证，证明OCA模块及其变体在多个计算机视觉任务中的优越性，
             尤其是在提高模型的效率和准确性方面。
通过正交化特征和注意力机制的结合，改进卷积神经网络在多尺度、多维度特征提取上的表现，提升计算机视觉任务中的性能与计算效率。

'''
__all__=['C2f_OCA1','C2f_OCA2']
from torch import Tensor
def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))
class GramSchmidtTransform(torch.nn.Module):
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, FWT: GramSchmidtTransform, input: Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class OCA1(nn.Module):
    def __init__(self, inplanes,planes, height, stride=1, downsample=None):
        super(OCA1, self).__init__()
        self._process: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.conv1x1 = nn.Conv2d(inplanes*4,inplanes,1)
        self.downsample = downsample
        self.stride = stride

        self.planes = planes
        self._excitation = nn.Sequential(
            nn.Linear(in_features=4 * planes, out_features=round(planes / 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 4), out_features=4 * planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention =Attention()
        self.F_C_A = GramSchmidtTransform.build(4 * planes, height)
    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0), out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attention = attention * out
        attention = self.conv1x1(attention)
        attention += residual
        activated = torch.relu(attention)
        return activated

class OCA2(nn.Module):
    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(OCA2, self).__init__()

        self._preprocess: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self._scale: nn.Module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        preprocess_out = self._preprocess(x)
        compressed = self.OrthoAttention(self.F_C_A, preprocess_out)
        b, c = preprocess_out.size(0), preprocess_out.size(1)
        attention = self._excitation(compressed).view(b, c, 1, 1)
        attentended = attention * preprocess_out
        scale_out = self._scale(attentended)
        scale_out += residual
        activated = torch.relu(scale_out)

        return activated
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


class Bottleneck_OCA1(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, height,shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = OCA1(c2,c2,height)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_OCA1(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False,height=64, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_OCA1(self.c, self.c,height, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

class Bottleneck_OCA2(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, height,shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = OCA1(c1,c2,height)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C2f_OCA2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False,height=64, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_OCA2(self.c, self.c,height, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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

# 输入 B C H W   输出 B C H W
if __name__ == '__main__':
    # 创建输入张量
    input = torch.randn(1, 64, 32, 32)
    # 定义 BasicBlock 模块
    block = C2f_OCA2(64,64,1,True, height=32)
    # 前向传播
    output = block(input)
    # 打印输入和输出的形状
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")