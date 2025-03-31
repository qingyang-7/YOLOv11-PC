import torch
import torch.nn as nn
import itertools
from timm.models.layers import DropPath
# 代码：https://github.com/xyongLu/SBCFormer/blob/master/models/sbcformer.py
# 论文：https://arxiv.org/pdf/2311.03747
__all__=['C2f_SBCFBlock',]
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

        # global FLOPS_COUNTER
        # output_points = ((resolution + 2 * padding - dilation *
        #                   (ks - 1) - 1) // stride + 1)**2
        # FLOPS_COUNTER += a * b * output_points * (ks**2) // groups

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 将 hidden_features 强制转换为整数
        self.fc1 = nn.Linear(in_features, int(hidden_features))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(hidden_features), out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class InvertResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.pwconv1_bn = Conv2d_BN(self.in_features, self.hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv_bn = Conv2d_BN(self.hidden_features, self.hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=self.hidden_features)
        self.pwconv2_bn = Conv2d_BN(self.hidden_features, self.in_features, kernel_size=1, stride=1, padding=0)

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    # @line_profile
    def forward(self, x):
        x1 = self.pwconv1_bn(x)
        x1 = self.act(x1)
        x1 = self.dwconv_bn(x1)
        x1 = self.act(x1)
        x1 = self.pwconv2_bn(x1)

        return x + x1

class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=2, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5

        self.nh_kd = key_dim * num_heads
        self.qk_dim = 2 * self.nh_kd
        self.v_dim = int(attn_ratio * key_dim) * num_heads
        dim_h = self.v_dim + self.qk_dim

        self.N = resolution ** 2
        self.N2 = self.N
        self.pwconv = nn.Conv2d(dim, dim_h, kernel_size=1, stride=1, padding=0)
        self.dwconv = Conv2d_BN(self.v_dim, self.v_dim, kernel_size=3, stride=1, padding=1, groups=self.v_dim)
        self.proj_out = nn.Linear(self.v_dim, dim)
        self.act = nn.GELU()

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        h, w = self.resolution, self.resolution
        # 正确的 reshape
        x = x.transpose(1, 2).reshape(B, C, h, w)

        x = self.pwconv(x)
        qk, v1 = x.split([self.qk_dim, self.v_dim], dim=1)
        qk = qk.reshape(B, 2, self.num_heads, self.key_dim, N).permute(1, 0, 2, 4, 3)
        q, k = qk[0], qk[1]

        v1 = v1 + self.act(self.dwconv(v1))
        v = v1.reshape(B, self.num_heads, -1, N).permute(0, 1, 3, 2)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_dim)
        x = self.proj_out(x)
        return x

class ModifiedTransformer(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, resolution=7):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=self.dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                              resolution=resolution)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim * mlp_ratio, out_features=self.dim,
                       act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBCFormerBlock(nn.Module):  # building block
    def __init__(self, dim, resolution=7,depth_invres=2, depth_mattn=1, depth_mixer=2, key_dim=16, num_heads=8, mlp_ratio=4., attn_ratio=2,
                 drop=0., attn_drop=0.,drop_paths=[0.2], pool_ratio=1, invres_ratio=1,):
        super().__init__()
        self.resolution = resolution
        self.dim = dim
        self.depth_invres = depth_invres
        self.depth_mattn = depth_mattn
        self.depth_mixer = depth_mixer
        self.act = h_sigmoid()

        self.invres_blocks = nn.Sequential()
        for k in range(self.depth_invres):
            self.invres_blocks.add_module("InvRes_{0}".format(k),
                                          InvertResidualBlock(in_features=dim, hidden_features=int(dim * invres_ratio),
                                                              out_features=dim, kernel_size=3, drop_path=0.))

        self.pool_ratio = pool_ratio
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
            self.convTrans = nn.ConvTranspose2d(dim, dim, kernel_size=pool_ratio, stride=pool_ratio, groups=dim)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.pool = nn.Identity()
            self.convTrans = nn.Identity()
            self.norm = nn.Identity()

        self.mixer = nn.Sequential()
        for k in range(self.depth_mixer):
            self.mixer.add_module("Mixer_{0}".format(k),
                                  InvertResidualBlock(in_features=dim, hidden_features=dim * 2, out_features=dim,
                                                      kernel_size=3, drop_path=0.))

        self.trans_blocks = nn.Sequential()
        for k in range(self.depth_mattn):
            self.trans_blocks.add_module("MAttn_{0}".format(k),
                                         ModifiedTransformer(dim=dim, key_dim=key_dim, num_heads=num_heads,
                                                             mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                                                             drop=drop, attn_drop=attn_drop, drop_path=drop_paths[k],
                                                             resolution=resolution))

        self.proj = Conv2d_BN(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_fuse = Conv2d_BN(self.dim * 2, self.dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, _, _ = x.shape
        h, w = self.resolution, self.resolution
        x = self.invres_blocks(x)
        local_fea = x

        if self.pool_ratio > 1.:
            x = self.pool(x)

        x = self.mixer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.trans_blocks(x)
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if self.pool_ratio > 1:
            x = self.convTrans(x)
            x = self.norm(x)
        global_act = self.act(self.proj(x))
        x_ = local_fea * global_act
        x_cat = torch.cat((x, x_), dim=1)
        out = self.proj_fuse(x_cat)

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
class C2f_SBCFBlock(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False,resolution=32, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SBCFormerBlock(self.c,resolution=resolution) for _ in range(n))

if __name__ == "__main__":
    import torch
    # 实例化 SBCFormerBlock
    SBCFBlock = C2f_SBCFBlock(64,64,1,False,resolution=32) #注意： 输入特征图的分辨率resolution=H=W
    # 创建一个随机的输入张量，形状为 B C H W
    input = torch.randn(2, 64, 32, 32)

    # 将输入张量通过 SBCFormerBlock
    output = SBCFBlock(input)

    # 打印输出张量的形状
    print(f"输入形状: {output.shape}")
    print(f"输出形状: {output.shape}")
