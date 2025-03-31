import torch
import torch.nn as nn
# 题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
# 论文地址：https://ieeexplore.ieee.org/document/10547405
__all__=['BFAM','CBM','MEDM']
'''
BFAM（双时相特征聚合模块）
作用：
BFAM的主要作用是聚合多个感受野的特征并补充纹理信息，从而增强变化检测网络对变化区域内部细节的捕捉能力。
它通过将低层纹理信息和高层语义信息相结合，提高了变化检测的完整性和准确性。
原理：
多感受野特征提取：BFAM使用四个并行的空洞卷积（dilated convolution）来提取不同感受野下的特征。
每个空洞卷积的空洞率不同（如1, 2, 3, 4），从而能够捕捉到不同尺度的变化区域。
特征拼接与降维：将四个不同感受野下提取的特征进行通道拼接，并通过1×1卷积进行降维，得到聚合后的低层纹理特征。
高层语义信息指导：使用SimAM注意力机制对高层语义特征进行处理，并将处理后的特征与低层纹理特征相乘，以融合高层语义信息。
残差连接：将聚合后的特征与前一阶段的特征通过残差连接进行融合，确保信息的完整性。

CBM（变化边界感知模块）
作用：
CBM的主要作用是捕捉变化区域的边界信息，并通过增强边界特征来指导后续的特征提取和变化检测过程。
它有助于减少噪声的影响，并提供丰富的上下文信息，以提高变化边界的准确性。

原理：
边界特征提取：CBM首先对输入特征进行SimAM注意力机制处理，突出重要区域。然后通过池化、相减和卷积操作提取边缘特征。
这些操作有助于突出边缘区域，抑制非边缘区域。
边界特征增强：将提取的边缘特征与原始特征进行相乘和相加操作，进一步增强边界特征的表达。
特征差异计算：对增强后的双时相特征进行差异计算，得到边缘增强的差异特征。这些差异特征包含了更准确的变化边界信息。
监督学习：对CBM分支的输出进行监督学习，通过计算预测结果与真实标签之间的差异来优化网络参数。这种监督学习有助于CBM更准确地捕捉变化边界。
'''
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
class BFAM(nn.Module):
    def __init__(self,inp):
        super(BFAM, self).__init__()

        self.pre_siam = simam_module()
        self.lat_siam = simam_module()
        out_1 = inp
        out = inp
        inp = inp + out
        self.conv_1 = nn.Conv2d(inp, out_1 , padding=1, kernel_size=3,groups=out_1,
                                   dilation=1)
        self.conv_2 = nn.Conv2d(inp, out_1, padding=2, kernel_size=3,groups=out_1,
                                   dilation=2)
        self.conv_3 = nn.Conv2d(inp, out_1, padding=3, kernel_size=3,groups=out_1,
                                   dilation=3)
        self.conv_4 = nn.Conv2d(inp, out_1, padding=4, kernel_size=3,groups=out_1,
                                   dilation=4)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_1 * 4, out_1, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_1),
            nn.ReLU(inplace=True)
        )

        self.fuse_siam = simam_module()

        self.out = nn.Sequential(
            nn.Conv2d(out_1, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self,data):
        inp1, inp2 =data[0],data[1]
        last_feature = None
        x = torch.cat([inp1,inp2],dim=1)
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1,c2,c3,c4],dim=1)
        fuse = self.fuse(cat)
        inp1_siam = self.pre_siam(inp1)
        inp2_siam = self.lat_siam(inp2)


        inp1_mul = torch.mul(inp1_siam,fuse)
        inp2_mul = torch.mul(inp2_siam,fuse)
        fuse = self.fuse_siam(fuse)
        if last_feature is None:
            out = self.out(fuse + inp1 + inp2 + inp2_mul + inp1_mul)
        else:
            out = self.out(fuse+inp2_mul+inp1_mul+last_feature+inp1+inp2)
        out = self.fuse_siam(out)

        return out
class diff_moudel(nn.Module):
    def __init__(self,in_channel):
        super(diff_moudel, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.simam = simam_module()
    def forward(self, x):
        x = self.simam(x)
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # weight = self.conv_1(edge)
        out = weight * x + x
        out = self.simam(out)
        return out
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

class CBM(nn.Module):
    def __init__(self,in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        self.simam = simam_module()
    def forward(self,data):
        x1, x2 = data[0],data[1]
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d

class MEDM(nn.Module):
    def __init__(self,in_channel):
        super(MEDM, self).__init__()
        self.diff_1 = MEEM(in_channel,in_channel)
        self.diff_2 = MEEM(in_channel,in_channel)
        self.simam = simam_module()
    def forward(self,data):
        x1, x2 = data[0],data[1]
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d

if __name__ == "__main__":
    input1 = torch.randn(1, 30, 128, 128)
    input2 = torch.randn(1, 30, 128, 128)
    bfam = BFAM(30)
    output = bfam([input1,input2])
    print('BFAM_input_size:', input1.size())
    print('BFAM_output_size:', output.size())
    cbm = CBM(30)
    output = cbm([input1,input2])
    print('CBM_input_size:', input1.size())
    print('CBM_output_size:', output.size())

    medm = MEDM(30)
    output = medm([input1,input2])
    print('medm_input_size:', input1.size())
    print('medm_output_size:', output.size())
