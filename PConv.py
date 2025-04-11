import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import C3

__all__=['PSConv','C2f_PSConv','C3k2_PSConv']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        # Adjust the kernel size for dilation if dilation is greater than 1
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        # If padding is not provided, calculate it automatically based on kernel size
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation function

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # Initialize the convolutional layer
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # Initialize batch normalization
        self.bn = nn.BatchNorm2d(c2)
        # Set the activation function based on the input argument
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization, and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # First convolution
        self.cv1 = Conv(c1, c_, k[0], 1)
        # Second convolution
        self.cv2 = Conv(c_, c2, k[1], 1, g)
        # Add shortcut if channels match and shortcut is enabled
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class PSConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        # Define 4 asymmetric padding methods for pinwheel-shaped convolution
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]  # Each tuple represents (left, top, right, bottom) padding
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]  # Create 4 padding layers

        # Horizontal convolution with kernel size (1, k), stride s, and output channels c2 // 4
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)

        # Vertical convolution with kernel size (k, 1), stride s, and output channels c2 // 4
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        # Final convolution layer to merge the results, kernel size (2, 2), output channels c2
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # Apply different padding and convolutions to get features from 4 directions
        yw0 = self.cw(self.pad[0](x))  # Horizontal direction, first padding
        yw1 = self.cw(self.pad[1](x))  # Horizontal direction, second padding
        yh0 = self.ch(self.pad[2](x))  # Vertical direction, first padding
        yh1 = self.ch(self.pad[3](x))  # Vertical direction, second padding

        # Concatenate the results along the channel dimension and process them through final convolution layer
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))  # Concatenate along channels and process with cat layer

class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and expansion."""
        super().__init__()

        # Use PSConv for the asymmetric padding convolutions
        self.psconv = PSConv(c1,c2)
        # Add shortcut if channels match and shortcut is enabled
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.psconv(x) if self.add else self.psconv(x)

class C2f_PSConv(nn.Module):
    """Faster Implementation of APCSP Bottleneck with Asymmetric Padding convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, P=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        # First convolution with kernel size (1, 1)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # Second convolution with kernel size (1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # Conditional creation of APBottleneck or Bottleneck layers based on P flag
        if P:
            self.m = nn.ModuleList(
                APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split the result into two halves along the channel dimension
        # Apply each bottleneck to the last chunk of the split data
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))  # Concatenate and pass through the final convolution

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))  # Split the result into two parts along the channel dimension
        # Apply each bottleneck to the last part of the split data
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))  # Concatenate and pass through the final convolution

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # Use APBottleneck with kernel size (3, 3) for each layer
        self.m = nn.Sequential(*(APBottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

class C3k2_PSConv(C2f_PSConv):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # Conditional inclusion of C3k or APBottleneck for each layer based on c3k flag
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

# Input: B C H W, Output: B C H W
if __name__ == "__main__":
    module = PSConv(c1=64, c2=128, k=3, s=2)  # Initialize PSConv module with specified parameters
    input_tensor = torch.randn(1, 64, 128, 128)  # Create a random input tensor with shape (B, C, H, W)
    output_tensor = module(input_tensor)  # Get the output after applying PSConv on input_tensor
    print('Input size:', input_tensor.size())  # Print the shape of input tensor
    print('Output size:', output_tensor.size())  # Print the shape of output tensor
