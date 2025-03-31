# --------------------------------------------------------
# 论文:HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection (arxiv 2024)
# github地址:https://github.com/zhengshuchen/HCFNet
# ------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
红外小物体检测是一项重要的计算机视觉任务，涉及红外图像中微小物体的识别和定位，红外图像通常只包含几个像素。
然而，由于物体的尺寸很小，而且红外图像的背景通常很复杂，因此它遇到了困难。
在本文中，我们提出了一种深度学习方法HCF-Net，该方法通过多个实用模块显著提高了红外小目标检测性能。
具体来说，它包括并行化贴片感知注意力 （PPA） 模块、维度感知选择性集成 （DASI） 模块和多扩张通道细化器 （MDCR） 模块。
PPA 模块使用多分支特征提取策略来捕获不同规模和级别的特征信息。
DASI模块支持自适应通道选择和融合。
MDCR模块通过多个深度可分离的卷积层捕获不同感受野范围的空间特征。
在SIRST红外单帧图像数据集上的大量实验结果表明，所提出的HCF-Net性能良好，超越了其他传统和深度学习模型。
'''
__all__=['DASI']
class DASI(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag()
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)
         self.skips_2 = conv_block(in_features=in_features * 2,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0),
                                 norm_type=None,
                                 activation=False)
         self.skips_3 = nn.Conv2d(in_features//2, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()
    # def forward(self, x, x_low, x_high):
    def forward(self,data):
        x, x_low, x_high = data[0],data[1],data[2]

        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.relu(x)

        return x
class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

# 输入 B C H W ,  输出 B C H W
if __name__ == '__main__':
    x = torch.randn(1, 128, 32, 32)  # 主输入特征
    x_low = torch.randn(1, 256, 16, 16)  # 低分辨率特征
    x_high = torch.randn(1, 64, 64, 64)  # 高分辨率特征

    # 初始化DASI模块
    dasi_module = DASI(in_features=128, out_features=256)
    data = [x, x_low, x_high]

    # 执行前向传播
    output = dasi_module(data)

    # 打印输出张量的形状
    print("输出的形状:", output.shape)
