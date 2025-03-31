import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = True
# 论文：https://arxiv.org/pdf/2406.03702
# 代码：https://github.com/takaniwa/DSNet
__all__=['MSAF_2D']
'''
即插即用模块：
MSAF_2D：多尺度注意力融合模块   CV2维任务通用模块
MSAF_3D：多尺度注意力融合模块   CV3维任务通用模块

MSAF是一种在DSNet中的多尺度注意力融合模块。
其核心思想是通过注意力机制来平衡不同尺度的上下文信息和细节信息，
并根据Sigmoid获取特征权重，实现选择性地融合来自不同尺度分支的信息。

接下来对MSAF模块简单介绍一下：
# 多尺度注意力（MSA）：
        该部分用于学习特征图中每个区域和像素的重要性，
        分别从区域注意力和像素注意力两个方面进行特征加权。
        通过对特征图进行池化和膨胀卷积操作，MSA模块能够
        在不同尺度上对特征进行压缩和扩展，从而适应不同的感受野需求。
# 多尺度注意力融合模块（MSAF）：
    在MSA模块中生成的特征权重将用于多尺度融合。
    MSAF模块通过加权的方式将上下文分支（Context Branch）和空间分支（Spatial Branch）的特征进行加权融合，
    结合多尺度的上下文信息和空间细节信息，从而提升语义分割任务中的特征表达能力。
适用于：语义分割，实例分割，目标检测，图像增强，暗光增强等所有CV任务通用的即插即用模块
'''
# MSAF
class MSAF_2D(nn.Module):

    def __init__(self, channels=64, r=4):
        super(MSAF_2D, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,data):
        x, y = data[0],data[1]
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + y
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo
class MSAF_3D(nn.Module):

    def __init__(self, channels=64, r=4):
        super(MSAF_3D, self).__init__()
        inter_channels = int(channels // r)

        # 修改为3D卷积和池化
        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool3d((8, 8, 8)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool3d((16, 16, 16)),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        d, h, w = x.shape[2], x.shape[3], x.shape[4]  # 获取输入 x 的深度、高度和宽度

        xa = x + y
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小
        c1 = F.interpolate(c1, size=[d, h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[d, h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[d, h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,
                                  padding=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
class MFACB(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2,2,2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes *3, out_planes, kernel_size=1, padding=0, bias=False ),
            BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # out1 = self.conv_list[0](x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1

# 输入 N C D H W,  输出 N C D H W
if __name__ == '__main__':
    block = MSAF_2D(channels=32, r=4)  # 输入通道数，r 是缩放比例
    x = torch.rand(4, 32, 64, 64)  # 输入2D张量：N C H W
    y = torch.rand(4, 32, 64, 64)  # 假设有residual连接
    output = block([x, y])
    print('2D_input size:', x.size())
    print('2D_output size:', output.size())


