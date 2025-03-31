from functools import partial

import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
__all__ = ['EUCB','MEUM']
# 论文地址：https://arxiv.org/abs/2405.06880
# 代码地址论文里面有
'''
来自CVPR 2024 顶会论文                        YOLOv8v10v11创新改进200+改进点商品 和 即插即用模块商品在评论区
即插即用模块：EUCB 有效上采样卷积模块 
      论文其中MSCAM和LGAG这两个模块我已经在交流群里面
      
提供二次创新模块 MEUM 多尺度增强上采样模块  效果优于EUCB，二次创新模块可以直接拿去发小论文，冲SCI一区  

EUCB模块（Efficient Up-Convolution Block）的主要作用是高效地进行上采样，
并增强特征图，以便与下一阶段的特征图进行融合。这对于提高解码器的性能和效率至关重要。
EUCB模块通过使用深度卷积代替传统3×3卷积，显著降低了计算复杂度，同时保持了特征图的上下文关系。
这使得解码器能够在低计算成本的情况下实现高效的多阶段特征聚合和精炼，尤其适用于资源有限的医疗图像分割场景。

EUCB模块原理：
上采样：EUCB首先通过上采样（Up()）将当前阶段的特征图分辨率提高。 作用：使得特征图尺寸扩大原来的2倍。
深度卷积：对上采样后的特征图应用一个3×3的深度卷积（DWC()）。 作用：以保留空间关系并减少计算复杂度。
特征增强：卷积操作之后，通过批归一化（BN()）和ReLU激活函数（ReLU()） 作用：引入非线性，同时避免梯度消失。
通道调整：最后使用1×1的卷积（C1×1()） 作用：调整特征图的通道数量，促进通道之间的信息交互。

这个即插即用模块适用于所有计算机视觉CV任务。
'''

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class MEEM(nn.Module):
    def __init__(self, in_dim, out_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        hidden_dim = in_dim // 2
        self.out = out_dim
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
            nn.Conv2d(hidden_dim * width, out_dim, 1, bias=False),
            norm(out_dim),
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

# 二次创新模块 MEUM 多尺度增强上采样模块  可以直接冲sci一区
'''
MEUM 多尺度增强上采样模块 内容介绍：

首先通过上采样（Up()）将当前阶段的特征图分辨率提高，使用一个比例因子（通常为2）扩大特征图的尺寸。
解决上采样后特征图中细节特征边缘缺失的问题，提高模型捕捉物体边界的能力。
使用3×3平均池化和1×1卷积从输入图像中提取多尺度的边缘信息。
通过边缘增强器（EE），在每个尺度上强化边缘感知，突出物体的关键边界。
提取的多尺度边缘信息与主分支的特征融合，从而增强输出特征图的特征。
'''
class MEUM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MEUM, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.FE = MEEM(in_dim=in_channels,out_dim=out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.FE(x)
        return x

if __name__ == '__main__':
    input = torch.randn(1,32, 64,64)
    EUCB = EUCB(32, 64) #输入通道数是32，输出通道数是64
    output = EUCB(input)
    print(f"EUCB_输入张量的形状: {input.size()}")
    print(f"EUCB_输入张量的形状: {output.size()}")

    MEUM = MEUM(32, 64)  #输入通道数是32，输出通道数是64
    output = MEUM(input)
    print(f"二次创新模块_MEUM_输入张量的形状: {input.size()}")
    print(f"二次创新模块_MEUM_输入张量的形状: {output.size()}")
