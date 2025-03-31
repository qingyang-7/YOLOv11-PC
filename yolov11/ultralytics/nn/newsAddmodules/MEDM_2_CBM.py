import torch
import torch.nn as nn
# 题目：B2CNet: A Progressive Change Boundary-to-Center Refinement Network for Multitemporal Remote Sensing Images Change Detection
# 论文地址：https://ieeexplore.ieee.org/document/10547405
__all__ = ['MEDM_2']
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
    def __init__(self,inp,out):
        super(BFAM, self).__init__()

        self.pre_siam = simam_module()
        self.lat_siam = simam_module()


        out_1 = inp
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

    def forward(self,inp1,inp2):
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
class CBM(nn.Module):
    def __init__(self,in_channel):
        super(CBM, self).__init__()
        self.diff_1 = diff_moudel(in_channel)
        self.diff_2 = diff_moudel(in_channel)
        self.simam = simam_module()
    def forward(self,x1,x2):
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d
# -----二次创新
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

class MEDM_2(nn.Module):
    def __init__(self,in_channel):
        super(MEDM_2, self).__init__()
        self.diff_1 = MEEM(in_channel,in_channel // 2)
        self.diff_2 = MEEM(in_channel,in_channel // 2)
        self.simam = simam_module()
    def forward(self,x):
        x1, x2 = x
        d1 = self.diff_1(x1)
        d2 = self.diff_2(x2)
        d = torch.abs(d1-d2)
        d = self.simam(d)
        return d

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    MEDM_module = MEDM_2(32)
    # 创建一个随机输入张量，假设批量大小为1，通道数为32，图像尺寸为64x64
    input1 = torch.randn(1, 32, 64, 64)
    input2 = torch.randn(1, 32, 64, 64)
    # 将输入张量传入 MEDM 模块
    output = MEDM_module(input1,input2)
    # 输出结果的形状
    print("输入张量的形状：", input1.shape)
    print("输出张量的形状：", output.shape)