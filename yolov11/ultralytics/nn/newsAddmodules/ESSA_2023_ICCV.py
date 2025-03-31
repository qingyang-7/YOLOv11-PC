import math
import torch
import torch.nn as nn
from ultralytics.nn.modules import C3
#论文： https://arxiv.org/pdf/2307.14010
__all__=['ESSAttn','C2f_ESSA','C3k2_ESSA']


'''
来自ICCV 2023 顶会论文
即插即用注意力模块： ESSA 有效自注意力模块

ESSA模块（Efficient SCC-kernel-based Self-Attention）是ESSAformer模型中的关键组件，
旨在提高高光谱图像超分辨率（HSI-SR）任务的计算效率和表现质量。
ESSA模块通过引入谱相关系数（SCC）计算特征之间的相似性，扩大感受野，捕获长距离的空间和光谱信息，
避免传统卷积网络因感受野限制而产生伪影。同时降低了计算复杂度并增强了模型在长距离依赖和小数据集上的表现，
特别适用于高光谱图像的超分辨率任务。它在提高图像恢复质量的同时，显著减少了计算负担，表现出色的视觉效果和定量结果。

ESSA模块的主要作用：
1.提高计算效率：通过核化自注意力机制，将传统自注意力计算的二次复杂度降低为线性复杂度,
             显著减少计算开销，尤其适用于高分辨率的高光谱图像。
2.增强长距离依赖建模：通过引入谱相关系数（SCC）计算特征之间的相似性，
           扩大感受野，捕获长距离的空间和光谱信息，避免传统卷积网络因感受野限制而产生的伪影。
3.提高数据效率与训练稳定性：SCC在计算注意力时，考虑了高光谱图像的光谱特性，
           能有效抗击阴影或遮挡等因素带来的光谱变化，提高模型在小数据集上的训练效率。
4.改善图像恢复质量：利用SCC的通道平移不变性和缩放不变性，减少由阴影、遮挡等造成的噪声，
           提高图像超分辨率恢复的质量，使得生成的高分辨率图像更加自然、平滑。
           
适用于：图像恢复，图像增强等所有计算机视觉CV任务通用的注意力模块

'''
class ESSAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lnqkv = nn.Linear(dim, dim * 3)
        self.ln = nn.Linear(dim, dim)

    def forward(self, x):
        b,c,h,w=x.shape
        x = x.reshape(b,c,h*w).permute(0,2,1)
        b, N, C = x.shape
        qkv = self.lnqkv(x)
        qkv = torch.split(qkv, C, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = torch.mean(q, dim=2, keepdim=True)
        q = q - a
        a = torch.mean(k, dim=2, keepdim=True)
        k = k - a
        q2 = torch.pow(q, 2)
        q2s = torch.sum(q2, dim=2, keepdim=True)
        k2 = torch.pow(k, 2)
        k2s = torch.sum(k2, dim=2, keepdim=True)
        t1 = v
        k2 = torch.nn.functional.normalize((k2 / (k2s + 1e-7)), dim=-2)
        q2 = torch.nn.functional.normalize((q2 / (q2s + 1e-7)), dim=-1)
        t2 = q2 @ (k2.transpose(-2, -1) @ v) / math.sqrt(N)

        attn = t1 + t2
        attn = self.ln(attn)
        x = attn.reshape(b,h,w,c).permute(0,3,1,2)
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

class Bottleneck_ESSA(nn.Module):
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
        self.Attention = ESSAttn(c2)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))



class C2f_ESSA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_ESSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
        self.m = nn.Sequential(*(Bottleneck_ESSA(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_ESSA(C2f_ESSA):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ESSA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    input =  torch.randn(1, 32, 64, 64)
    model = ESSAttn(32)
    output = model(input)
    print('input_size:',input.size())
    print('output_size:',output.size())
