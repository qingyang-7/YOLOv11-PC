from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
__all__=['HIFA','C2f_HIFA']
'''
SCI一区 2024
即插即用特征增强模块：HIFA    所有CV任务通用模块
HIFA模块是为了在医学图像分割任务中更好地结合局部与全局信息而设计的。
其核心目标是能够捕获低频信息（如全局结构和语义）以及高频信息（如局部边缘和纹理），
从而更好地模拟人类视觉系统处理图像的方式。

HIFA模块的核心组成：
空间金字塔池化（SPP）：用于捕获多尺度的全局上下文信息。这可以帮助模型在不同尺度下理解全局语义。
多尺度膨胀卷积：通过不同的膨胀率提取局部上下文信息，从而捕获更多细节如边缘和纹理。

作用：
HIFA模块能够有效融合局部细节和全局结构信息，提升分割模型的表现，特别是在复杂结构或不同尺度信息要求高的医学图像分割任务中。
通过这种全局与局部信息的增强与融合，HIFA模块可以实现更高效的特征编码与解码，提高分割精度。
HIFA模块通过整合全局和局部的特征信息，为网络提供更丰富的上下文信息，显著提升了分割任务中的表现。

'''

# ############################################## HIFA_module###########################################
def BNReLU(num_features):
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )

class SPP_inception_block(nn.Module):
    def __init__(self, in_channels):
        super(SPP_inception_block, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)  # [3, 3]
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)  # [2, 2]
        # self.pool = nn.MaxPool2d(kernel_size=[4, 4], stride=4) # [1, 1]
        # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=2) # [4, 4]
        # self.pool = nn.MaxPool2d(kernel_size=[1, 1], stride=1)   # [7, 7]
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()  # [4, 256, 7, 7]
        pool_1 = self.pool1(x).view(b, c, -1)  # [2, 256, 3, 3], [2, 256, 9]
        # pool_1 = self.pool(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)  # [2, 256, 2, 2], [2, 256, 4]
        pool_3 = self.pool3(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]
        pool_4 = self.pool4(x).view(b, c, -1)  # [2, 256, 1, 1], [2, 256, 1]

        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)  # [2, 256, 15]

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(
            self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))  # self.conv1x1 is not necessary

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out  # [2, 256, 7, 7]
        cnn_out = cnn_out.view(b, c, -1)  # [2, 256, 49]

        out = torch.cat([pool_cat, cnn_out], -1)  # [2, 256, 64]
        out = out.permute(0, 2, 1)  # [2, 64, 256]

        return out
class NonLocal_spp_inception_block(nn.Module):
    def __init__(self, in_channels=512, ratio=2):
        super(NonLocal_spp_inception_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = in_channels // ratio
        self.value_channels = in_channels // ratio

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            BNReLU(self.key_channels),
        )

        self.f_query = self.f_key

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.spp_inception_v = SPP_inception_block(self.key_channels)
        self.spp_inception_k = SPP_inception_block(self.key_channels)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)  # [2, 512, 7, 7]

        x_v = self.f_value(x)  # [2, 256, 7, 7]
        value = self.spp_inception_v(x_v)  # [2, 64, 256]  15+49

        query = self.f_query(x).view(batch_size, self.key_channels, -1)  # [2, 256, 7, 7], [2, 256, 49]
        query = query.permute(0, 2, 1)  # [2, 49, 256]

        x_k = self.f_key(x)  # [2, 256, 7, 7]
        key = self.spp_inception_k(x_k)  # [2, 64, 256]  15+49
        key = key.permute(0, 2, 1)  # # [2, 256, 64]

        sim_map = torch.matmul(query, key)  # [2, 49, 64]
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)  # [2, 49, 256]
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])  # [4, 256, 7, 7]
        context = self.W(context)  # [4, 512, 7, 7]
        return context

class HIFA(nn.Module):
    def __init__(self, in_channels=512, ratio=2, dropout=0.0):
        super(HIFA, self).__init__()

        self.NSIB = NonLocal_spp_inception_block(in_channels=in_channels, ratio=ratio)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            BNReLU(in_channels)
            # nn.Dropout2d(dropout)
        )
    def forward(self, feats):
        att = self.NSIB(feats)
        output = self.conv_bn_dropout(torch.cat([att, feats], 1))
        return output


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
class C2f_HIFA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(HIFA(self.c) for _ in range(n))

if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1, 512, 32, 32)
    # 创建一个HIFA实例
    HIFA = HIFA(in_channels=512)
    # 将输入特征图传递给 HIFA模块
    output = HIFA(input)
    # 打印输入和输出的尺寸
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")