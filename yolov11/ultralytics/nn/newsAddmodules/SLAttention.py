import torch
import torch.nn as nn
from einops import rearrange
#代码： https://github.com/xinghaochen/SLAB
#论文：https://arxiv.org/pdf/2405.11582
__all__=['SimplifiedLinearAttention','C2f_SLA']
'''
SLAB：具有简化线性注意力和渐进式重新参数化批量归一化的高效转换器   
                         ICML 2024国际顶会
简化线性注意力即插即用模块：SLAttention  特点：简单有效，捕获注意力特征信息的能力非常好

注意力计算成本对于transform的高效运行至关重要，而现有方法难以在效率和精度之间取得良好的平衡。
为此，我们提出了一种简化线性注意力（SLA）模块，该模块利用ReLU作为内核函数，
并结合深度卷积进行局部特征增强，所提出的注意力机制比先前的线性注意力更有效，并且获得了相当好的性能。

特点：简单有效，捕获注意力特征信息的能力非常好

适用于：图像分类任务，目标检测任务等所有CV2维任务和NLP任务通用的注意力模块。
'''

class SimplifiedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=(80,80), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        # self.dwc = nn.Sequential(nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1),
        #                          nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1)
        #                          )
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b1,c1,h1,w1 = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        x=x.reshape(b1, c1, -1).transpose(-1, -2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        # 这里调整 positional_encoding 的大小以匹配输入的大小 N
        positional_encoding = self.positional_encoding[:, :N, :]
        k = k + positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(b1, c1, h1, w1)
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

    def __init__(self, c1, c2, shortcut=True,window_size=(80,80),g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.SLA = SimplifiedLinearAttention(c2,window_size=window_size)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        x1 = self.cv2(self.cv1(x))
        x1 = self.SLA(x1)
        return x + x1 if self.add else x1
class C2f_SLA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False,window_size=(80,80),g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut,window_size,g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 80, 80)
    image = torch.rand(*image_size)

    # Model
    model = C2f_SLA(64, 64)

    out = model(image)
    print(len(out))
# if __name__ == '__main__':
#     window_size = (80, 80)  # 注意：设定窗口大小等于输入图片的 H,W
#     num_heads = 8  # 注意力头数
#     # 创建 SimplifiedLinearAttention 实例
#     SLA = SimplifiedLinearAttention(dim=64, window_size=window_size, num_heads=num_heads)
#
#     # 1.如何输入的是图片4维数据 . CV方向的小伙伴都可以拿去使用
#     # 随机生成输入4维度张量：B, C, H, W
#     input = torch.randn(1, 64, 80,80)
#     # input = input_img.reshape(1, 64, -1).transpose(-1, -2)  # B L C :1 1024 64
#     # # 运行前向传递
#     output = SLA(input)
#     # output_img = output.view(1, 64, 32, 32)  # 将三维度转化成图片四维度张量
#     # 输出输入图片张量和输出图片张量的形状
#     print("CV_SLA_input size:", input.size())
#     print("CV_SLA_Output size:", output.size())
#
#     # # 2.如何输入的3维数据 . NLP方向的小伙伴都可以拿去使用
#     # B, L, C = 1, 1024, 64   # 批量大小、序列长度、特征维度
#     # # 创建一个随机的输入三维张量
#     # input = torch.randn(B, L, C)  # 输入三维张量为 (B, L, C)
#     # # 进行前向传播
#     # output = SLA(input)
#     # print('NLP_SLA_input size:', input.size())
#     # print('NLP_SLA_output size:', output.size())