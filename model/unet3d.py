import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3x3x3 Conv + InstanceNorm + LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


## 标准的UNet
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8):
        super(UNet3D, self).__init__()

        # 编码器部分 (5层)
        self.enc1 = ConvBlock3D(in_channels, base_channels * 2**0)   # 8
        self.enc2 = ConvBlock3D(base_channels * 2**0, base_channels * 2**1)  # 16
        self.enc3 = ConvBlock3D(base_channels * 2**1, base_channels * 2**2)  # 32
        self.enc4 = ConvBlock3D(base_channels * 2**2, base_channels * 2**3)  # 64
        self.enc5 = ConvBlock3D(base_channels * 2**3, base_channels * 2**4)  # 128

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 解码器部分
        self.up4 = nn.Conv3d(base_channels * 2**4, base_channels * 2**3, kernel_size=1)
        self.dec4 = ConvBlock3D(base_channels * 2**4, base_channels * 2**3)

        self.up3 = nn.Conv3d(base_channels * 2**3, base_channels * 2**2, kernel_size=1)
        self.dec3 = ConvBlock3D(base_channels * 2**3, base_channels * 2**2)

        self.up2 = nn.Conv3d(base_channels * 2**2, base_channels * 2**1, kernel_size=1)
        self.dec2 = ConvBlock3D(base_channels * 2**2, base_channels * 2**1)

        self.up1 = nn.Conv3d(base_channels * 2**1, base_channels * 2**0, kernel_size=1)
        self.dec1 = ConvBlock3D(base_channels * 2**1, base_channels * 2**0)

        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # 解码器 (最近邻上采样 + Conv)
        d4 = F.interpolate(e5, scale_factor=2, mode='nearest')
        d4 = self.up4(d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, scale_factor=2, mode='nearest')
        d3 = self.up3(d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        # print('x shape: ',x.shape,'\n',
        # 'e1 shape: ',e1.shape,'\n',
        # 'e2 shape: ',e2.shape,'\n',
        # 'e3 shape: ',e3.shape,'\n',
        # 'e4 shape: ',e4.shape,'\n',
        # 'e5 shape: ',e5.shape,'\n',
        # 'd4 shape: ',d4.shape,'\n',
        # 'd3 shape: ',d3.shape,'\n',
        # 'd2 shape: ',d2.shape,'\n',
        # 'd1 shape: ',d1.shape,'\n',
        # 'out shape: ',out.shape,'\n')
        return out


## UNet++
class UNetPlusPlus3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8):
        super().__init__()

        filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16
        ]

        self.pool = nn.MaxPool3d(2, 2)

        # encoder (x{i,0})
        self.x0_0 = ConvBlock3D(in_channels, filters[0])
        self.x1_0 = ConvBlock3D(filters[0], filters[1])
        self.x2_0 = ConvBlock3D(filters[1], filters[2])
        self.x3_0 = ConvBlock3D(filters[2], filters[3])
        self.x4_0 = ConvBlock3D(filters[3], filters[4])

        # decoder (nested)
        self.x0_1 = ConvBlock3D(filters[0] + filters[1], filters[0])
        self.x1_1 = ConvBlock3D(filters[1] + filters[2], filters[1])
        self.x2_1 = ConvBlock3D(filters[2] + filters[3], filters[2])
        self.x3_1 = ConvBlock3D(filters[3] + filters[4], filters[3])

        self.x0_2 = ConvBlock3D(filters[0]*2 + filters[1], filters[0])
        self.x1_2 = ConvBlock3D(filters[1]*2 + filters[2], filters[1])
        self.x2_2 = ConvBlock3D(filters[2]*2 + filters[3], filters[2])

        self.x0_3 = ConvBlock3D(filters[0]*3 + filters[1], filters[0])
        self.x1_3 = ConvBlock3D(filters[1]*3 + filters[2], filters[1])

        self.x0_4 = ConvBlock3D(filters[0]*4 + filters[1], filters[0])

        # output
        self.out_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def up(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x):
        # encoder
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))

        # level 1
        x0_1 = self.x0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # level 2
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # level 3
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # level 4
        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        return self.out_conv(x0_4)



# 🔍 测试网络结构
if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(1, 1, 128, 128, 128)  # batch=1, channel=1, volume=128^3
    y = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", y.shape)
