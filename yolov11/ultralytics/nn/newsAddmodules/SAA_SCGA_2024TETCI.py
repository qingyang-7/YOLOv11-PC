import torch
import torch.nn.functional as F
from torch import nn
# 代码地址：https://github.com/YishuLiu/TransAttUnet/tree/main
# 论文地址：https://arxiv.org/pdf/2107.05274
__all__= ['SAA','SCGA','C2f_SAA','C3k2_SAA','C2f_SCGA','C3k2_SCGA']
from ultralytics.nn.modules import C3

'''
来自TETCI 2024 论文
即插即用注意力： SAA 自我感知注意力        
提供二次创新  SCGA 自我感知协调注意力 效果优于SAA,可以直接拿去冲SCI一区  

从医学图像中精确分割器官或病变对疾病诊断和器官形态测量至关重要。
近年来，卷积编码器-解码器结构在自动医学图像分割领域取得了显著进展。
然而，由于卷积操作的固有偏差，现有模型主要关注由邻近像素形成的局部视觉线索，未能充分建模长程上下文依赖性。
本文提出了一种新颖的基于Transformer的注意力引导网络，称为 TransAttUnet。
该网络设计了多级引导注意力和多尺度跳跃连接，以共同增强语义分割架构的性能。
受Transformer启发，本文将 自感知注意力模块 (SAA) 融入TransAttUnet中，
该模块结合了Transformer自注意力 (TSA) 和全局空间注意力 (GSA)，能够有效地学习编码器特征之间的非局部交互。
此外，本文还在解码器块之间引入了多尺度跳跃连接，用于将不同语义尺度的上采样特征进行聚合，
从而增强多尺度上下文信息的表示能力，生成具有区分性的特征。得益于这些互补组件，
TransAttUnet能够有效缓解卷积层堆叠和连续采样操作引起的细节丢失问题，最终提升医学图像分割的质量。
在多个医学图像分割数据集上的大量实验表明，所提出的方法在不同成像模式下始终优于最新的基线模型。

SAA 模块是作用在于增强医学图像分割的上下文语义建模能力和全局空间关系表征能力。
其核心由以下两部分组成：
1.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。

2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。

适用于：医学图像分割，目标检测，语义分割，图像增强，暗光增强，遥感图像任务等所有计算机视觉CV任务通用注意力模块
'''
class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class ScaledDotProductAttention(nn.Module):
    '''自注意力模块'''

    def __init__(self, temperature=512, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        m_batchsize, d, height, width = x.size()
        q = x.view(m_batchsize, d, -1)
        k = x.view(m_batchsize, d, -1)
        k = k.permute(0, 2, 1)
        v = x.view(m_batchsize, d, -1)

        attn = torch.matmul(q / self.temperature, k)

        if mask is not None:
            # 给需要mask的地方设置一个负无穷
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        output = output.view(m_batchsize, d, height, width)

        return output
class SAA(nn.Module):
    def __init__(self, in_channels):
        super(SAA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        self.tsa = ScaledDotProductAttention()
    def forward(self, x):
        x1 = self.gsa(x)
        x2 = self.gsa(x)
        out = x1 + x2
        return out
# 二次创新注意力模块 SCGA 自我感知协调注意力 冲SCI一区
'''
SCGA 自我感知协调注意力 内容介绍：

1.执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。这种方法避免了降维问题，
确保模型能够有效地聚焦在最相关的通道特征上。
2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。
3.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。
'''
class SCGA(nn.Module):
    def __init__(self, in_channels):
        super(SCGA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        self.tsa = ScaledDotProductAttention()
        self.ca = ChannelAttention(in_channels)
    def forward(self, x):
        x1 = x * self.ca(x)
        x1 = x1 * self.gsa(x)
        x2 = self.tsa(x)
        out = x1 + x2
        return out
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
class Bottleneck_SAA(nn.Module):
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
        self.Attention = SAA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
class Bottleneck_SCGA(nn.Module):
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
        self.Attention = SCGA(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))
class C2f_SAA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_SAA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
class C2f_SCGA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_SCGA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
        self.m = nn.Sequential(*(Bottleneck_SAA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
class C3k_SCGA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SCGA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_SAA(C2f_SAA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SAA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
class C3k2_SCGA(C2f_SCGA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_SCGA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SAA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.rand(1, 64, 128, 128)
    SAA = SAA(in_channels=64)
    output = SAA(input)
    print("SAA_input.shape:", input.shape)
    print("SAA_output.shape:",output.shape)
    SCGA = SCGA(in_channels=64)
    output = SCGA(input)
    print("二次创新_SCGA_input.shape:", input.shape)
    print("二次创新_SCGA_output.shape:",output.shape)

