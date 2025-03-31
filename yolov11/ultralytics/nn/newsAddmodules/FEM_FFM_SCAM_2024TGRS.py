import math
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
# 论文地址：https://ieeexplore.ieee.org/document/10423050
# 代码地址在论文里面
__all__ = ['FEM','FFM_Concat3','FFM_Concat2','SCAM','MSFE',]

'''
# 来自TGRS 2024  一区 小目标检测任务论文  采用的是YOLO模型发的2024年TGRS顶刊 
即插即用模块：FEM 特征增强模块 、FFM 特征融合模块 、 SCAM 空间上下文感知模块
含二次创新模块  MSFE 多尺度特征增强模块  MSFE是对FEM的二次创新，效果优于FEM模块,可以直接拿去发小论文  冲sci一区或二区
            
# 手把手教大家学会写顶刊/会摘要     往往写好一篇论文摘要，会直接影响你的文章是否能中稿。或是直接拒稿
由于特征表示不足和背景混淆等问题，使得遥感图像中的小目标检测任务变得极为困难。
                                           ---提出问题，交代本文任务（做遥感图像小目标检测任务）

特别是当算法需在进行实时处理时，需要在有限的计算资源下对精度和速度进行广泛优化。
为了解决这些问题，本文提出了一种高效的检测器，称为“特征增强、融合与上下文感知YOLO”。
                                           ---根据以上问题，引出本文创新点

本文YOLO包含三个创新的轻量化即插即用模块：特征增强模块（FEM）、特征融合模块（FFM）和空间上下文感知模块（SCAM）。
这三个模块分别提升了网络的局部区域感知能力、多尺度特征融合能力，以及跨通道和空间的全局关联能力，
同时尽量避免了复杂度的增加。通过这些改进，弱特征表达的小目标得到了增强，而易混淆的背景被有效抑制。
                                           ---对本文创新点，进行简单介绍

本文使用两个公开的遥感小目标检测数据集（VEDAI和AI-TOD）以及一个自建数据集（USOD）对本文YOLO方法的有效性进行了验证。
本文方法在mAP50指标上的精度分别达到0.748、0.617和0.909，超过了多个基准模型和最新方法。
同时，在不同的模拟退化条件下，其鲁棒性也得到了验证。   
 ---通过广泛实验验证本文方法是有效性（最后一句要包含：在那几个数据集上实验，本文方法在评价指标上效果怎样【列数字/写百分比%】,
                                超过了多个基准模型和最新方法,我们希望对今后在xx方面的研究提供一些启发等等）


特征增强模块（FEM）作用及原理：
作用：增强小目标的特征表达能力，通过引入多分支空洞卷积结构，获取更丰富的局部上下文信息，
同时扩大感受野，增强对复杂背景的抑制能力。
原理：FEM采用多分支卷积结构，其中包括标准卷积和空洞卷积，通过级联卷积操作提取特征。
空洞卷积能够保留更多的上下文信息，而多分支结构提供多样化的语义信息输出，同时保持较低的参数量。

特征融合模块（FFM）作用及原理：
作用：优化多尺度特征融合，提升对多层次语义信息的表示能力，同时降低复杂背景的干扰。
原理：基于BiFPN改进的结构，通过通道权重机制（CRC）对不同尺度的特征图进行通道级别的加权重构。
FFM设计了自上而下和自下而上的特征流动策略，实现浅层到深层、深层到浅层的语义信息交互。

空间上下文感知模块（SCAM）作用及原理：
作用：通过全局上下文关系建模，增强特征的全局表征能力，抑制背景噪声，提高小目标的区分性。
原理：SCAM通过全局平均池化（GAP）和全局最大池化（GMP）提取全局信息，
并使用轻量化的查询-键值（Query-Key）机制建模空间和通道间的上下文交互,
最终实现通道和空间上下文信息的融合。

适用于：小目标检测任务，小目标分割任务，图像增强任务，图像分类任务，暗光增强任务
       超分图像任务，遥感图像任务,图像恢复等所有计算机视觉CV任务通用的即插即用模块
'''

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv_withoutBN(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class SCAM(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SCAM, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels

        self.k = Conv(in_channels, 1, 1, 1)
        self.v = Conv(in_channels, self.inter_channels, 1, 1)
        self.m = Conv_withoutBN(self.inter_channels, in_channels, 1, 1)
        self.m2 = Conv(2, 1, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP

    def forward(self, x):
        n, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # avg max: [N, C, 1, 1]
        avg = self.avg_pool(x).softmax(1).view(n, 1, 1, c)
        max = self.max_pool(x).softmax(1).view(n, 1, 1, c)

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)

        # y2:[N, 1, H, W]
        y_avg = torch.matmul(avg, v).view(n, 1, h, w)
        y_max = torch.matmul(max, v).view(n, 1, h, w)

        # y_cat:[N, 2, H, W]
        y_cat = torch.cat((y_avg, y_max), 1)

        y = self.m(y) * self.m2(y_cat).sigmoid()

        return x + y


class FFM_Concat2(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1):
        super(FFM_Concat2, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel_all = int(Channel1 + Channel2)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型 parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()

        w = self.w[:(C1 + C2)] # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x = [x1, x2]
        return torch.cat(x, self.d)

class FFM_Concat3(nn.Module):
    def __init__(self, dimension=1, Channel1 = 1, Channel2 = 1, Channel3 = 1):
        super(FFM_Concat3, self).__init__()
        self.d = dimension
        self.Channel1 = Channel1
        self.Channel2 = Channel2
        self.Channel3 = Channel3
        self.Channel_all = int(Channel1 + Channel2 + Channel3)
        self.w = nn.Parameter(torch.ones(self.Channel_all, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        N1, C1, H1, W1 = x[0].size()
        N2, C2, H2, W2 = x[1].size()
        N3, C3, H3, W3 = x[2].size()

        w = self.w[:(C1 + C2 + C3)]  # 加了这一行可以确保能够剪枝
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion

        x1 = (weight[:C1] * x[0].view(N1, H1, W1, C1)).view(N1, C1, H1, W1)
        x2 = (weight[C1:(C1 + C2)] * x[1].view(N2, H2, W2, C2)).view(N2, C2, H2, W2)
        x3 = (weight[(C1 + C2):] * x[2].view(N3, H3, W3, C3)).view(N3, C3, H3, W3)
        x = [x1, x2, x3]
        return torch.cat(x, self.d)


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
# 二次创新：MSFE  可以直接拿去发小论文，冲sci 一区或二区
'''
MSFE 多尺度特征增强模块
MSFE 多尺度特征增强模块的内容介绍：
MSFE 多尺度特征增强模块
多尺度特征增强模块解决显著目标检测（SOD）中细节边缘缺失的问题，提高模型捕捉物体边界的能力。
使用3×3平均池化和1×1卷积从输入图像中提取多尺度的边缘信息。
通过边缘增强器（EE），在每个尺度上强化边缘感知，突出物体的关键边界。
提取的多尺度边缘信息与主分支的特征融合，提升最终预测结果的精细度。
增强小目标的特征表达能力，通过引入多分支空洞卷积结构，获取更丰富的局部上下文信息，
同时扩大感受野，增强对复杂背景的抑制能力。
MSFE采用多分支卷积结构，其中包括标准卷积和空洞卷积，通过级联卷积操作提取特征。
空洞卷积能够保留更多的上下文信息，而多分支结构提供多样化的语义信息输出，同时保持较低的参数量。
'''

class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge
class MSFE(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(MSFE, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.me = MEEM(out_planes,out_planes//2)
    def forward(self, x):
        # x = self.att(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out + short
        out = self.me(out)
        return out



# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    # 初始化 FEM 模块并设定通道维度
    FEM_module = FEM(32,64)
    # 创建一个随机输入张量，假设批量大小为1，通道数为32，图像尺寸为64x64
    input = torch.randn(1, 32, 64, 64)
    # 将输入张量传入 FEM 模块
    output = FEM_module(input)
    # 输出结果的形状
    print("FEM_输入张量的形状：", input.shape)
    print("FEM_输出张量的形状：", output.shape)

    # 初始化 MSFE 模块并设定通道维度
    MSFE_module = MSFE(32, 64)
    output = MSFE_module(input)
    # 输出结果的形状
    print("二次创新MSFE_输入张量的形状：", input.shape)
    print("二次创新MSFE_输出张量的形状：", output.shape)
