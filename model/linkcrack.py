import torch.nn.functional as F
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F


from einops.layers.torch import Rearrange

class IRDM(nn.Module):
    def __init__(self, ninput, noutput):
        super(IRDM,self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            
            self.conv = nn.Conv2d(ninput, noutput-ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            
            self.conv1 = nn.Conv2d(ninput, noutput // 2, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(noutput // 2, noutput, kernel_size=3, stride=2, padding=1)


        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv1(x)
            output = self.conv2(output)

        output = self.bn(output)
        return F.relu(output)
    


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

    def forward(self, x):
        
        upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(upsampled)

class DEABlock(nn.Module):
    def __init__(self, dim, kernel_size, reduction=8):
        super(DEABlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = res + x
        res = self.conv2(res)
        cattn = self.ca(res)
        sattn = self.sa(res)
        pattn1 = sattn + cattn
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        res = res + x
        return res

class LUM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, reduction=8):
        super(LUM, self).__init__()
        self.main_path = Upsample(in_channels, out_channels, kernel_size, stride)
        self.side_path = DEABlock(out_channels, kernel_size, reduction)

    def forward(self, x):
       
        main_out = self.main_path(x)
       
        side_out = self.side_path(main_out)
        
        return main_out + side_out



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bias=False, bn1=True,bn2=True):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5, momentum=0.01, affine=True) if bn1 else None
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn2 else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.depthwise(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.pointwise(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DSFEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(DSFEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            DepthwiseSeparableConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            DepthwiseSeparableConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1)
        )
        self.branch1 = nn.Sequential(
            DepthwiseSeparableConv(in_planes, inter_planes, kernel_size=1, stride=1),
            DepthwiseSeparableConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            DepthwiseSeparableConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            DepthwiseSeparableConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5)
        )
        self.branch2 = nn.Sequential(
            DepthwiseSeparableConv(in_planes, inter_planes, kernel_size=1, stride=1),
            DepthwiseSeparableConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            DepthwiseSeparableConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            DepthwiseSeparableConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        self.ConvLinear = DepthwiseSeparableConv(6 * inter_planes, out_planes, kernel_size=1, stride=1)
        self.shortcut = DepthwiseSeparableConv(in_planes, out_planes, kernel_size=1, stride=stride)
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


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=(1, 1), shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation[0], padding=dilation[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, dilation=dilation[1], padding=dilation[1], bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels,  BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        
        self.BN_enable = BN_enable
        

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                              bias=True)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.up = LUM(mid_channels,mid_channels)

    def forward(self, down_inp, up_inp):
        x = torch.cat([down_inp, up_inp], 1)
        x = self.conv1(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu(x)
       
        x = self.up(x)
        x = self.conv2(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu(x)
        return x


class LinkCrack(nn.Module):

    def __init__(self):
        super(LinkCrack, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self._res1_shorcut = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.res1 = nn.Sequential(
            
            ResidualBlock(64, 64, stride=1, shortcut=self._res1_shorcut),
            IRDM(64,64),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
        )


        self._res2_shorcut = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.res2 = nn.Sequential(
            IRDM(64,64),
            ResidualBlock(64, 64, stride=1, shortcut=self._res2_shorcut),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
            ResidualBlock(64, 64, ),
        )

        self._res3_shorcut = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

       
        self.res3 = nn.Sequential(
            IRDM(64,128),
            ResidualBlock(128, 128, stride=1, shortcut=self._res3_shorcut),
            ResidualBlock(128, 128), 
            DSFEM(in_planes=128, out_planes=128),
            ResidualBlock(128, 128, dilation=(2,2)), 
            DSFEM(in_planes=128, out_planes=128),
            ResidualBlock(128, 128, dilation=(2,2)),
        )

        self._res4_shorcut = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )

       
        self.res4 = nn.Sequential(
            ResidualBlock(128, 128,dilation=(2,2), shortcut=self._res4_shorcut),
            DSFEM(in_planes=128, out_planes=128),   
            ResidualBlock(128, 128, dilation=(4,4)),
        )
        self.dec4 = DecoderBlock(in_channels=128+128, mid_channels=128, out_channels=64)
        self.dec3 = DecoderBlock(in_channels=64+64, mid_channels=64, out_channels=64)
        self.dec2 = DecoderBlock(in_channels=64+64, mid_channels=64, out_channels=64)

        self.mask = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
        )

        self.link = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
        )

    def forward(self, x):
     
        x = self.pre(x)
        
        x1 = self.res1(x)
        
        x2 = self.res2(x1)
        
        x3 = self.res3(x2)
       
        x4 = self.res4(x3)
        
        x5 = self.dec4(x4,x3)
       
        x6 = self.dec3(x5, x2)
        
        x7 = self.dec2(x6,x1)
       
        mask = self.mask(x7)
        link = self.link(x7)
        return mask, link
