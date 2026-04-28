from torch import nn

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PromptEncoder, self).__init__()
        self.firstConv = DoubleConv(in_channel=in_channel, out_channel=out_channel//16)

        self.d1 = Down(in_channel=out_channel // 16, out_channel=out_channel // 8)
        self.d2 = Down(in_channel=out_channel // 8, out_channel=out_channel // 4)
        self.d3 = Down(in_channel=out_channel // 4, out_channel=out_channel // 2)
        self.d4 = Down(in_channel=out_channel // 2, out_channel=out_channel)

    def forward(self, x):
        # print('x.shape', x.shape)
        x1 = self.firstConv(x)
        # print('x1.shape', x1.shape)
        x2 = self.d1(x1)
        # print('x2.shape', x2.shape)
        x3 = self.d2(x2)
        # print('x3.shape', x3.shape)
        x4 = self.d3(x3)
        # print('x4.shape', x4.shape)
        x5 = self.d4(x4)
        # print('x5.shape', x5.shape)

        skip = [x4, x3, x2, x1]

        return x5, skip


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        mid_channel = in_channel//2 if in_channel > out_channel else out_channel//2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(mid_channel),
            nn.GELU(),

            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(out_channel),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel=in_channel, out_channel=out_channel)
        )

    def forward(self, x):
        return self.down(x)