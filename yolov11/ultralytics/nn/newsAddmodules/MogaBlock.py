import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import  DropPath

__all__=['C2f_MogaBlock']

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = nn.BatchNorm2d(embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
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
class C2f_MogaBlock(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MogaBlock(self.c) for _ in range(n))
