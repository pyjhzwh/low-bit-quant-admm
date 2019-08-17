#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample(data, outsize=28):
    data = F.interpolate(data,outsize, mode = 'bilinear')
    #print(data.size(),"=?",outsize)
    return data



class all_cnn_net(nn.Module):
    def __init__(self, block, layers, nclass=10, ds=False):
        super(ResNet,self).__init__()
        self.nclass = nclass
        self.ds = ds

        self.conv0 = ConvBlock(3, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv1 = ConvBlock(96, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv2 = ConvBlock(96, 96, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout0 = nn.Dropout(p=0.5)

        self.conv3 = ConvBlock(96, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv4 = ConvBlock(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv5 = ConvBlock(192, 192, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout1 = nn.Dropout(p=0.5)

        self.conv6 = ConvBlock(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv7 = ConvBlock(192, 192, kernel_size=1, padding=0, stride=1, relu=True)
        self.conv8 = ConvBlock(192, 10, kernel_size=1, padding=0, stride=1, relu=True)

        #self.avgpool = nn.AvgPool2d(kernel_size=6, stride=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.normal_(m.weight)
                #nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _conv_block(in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True):
        if relu:
            conv_block = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kern, stride=stride, bias=False),
                    nn.ReLU(in_planes = True)
                    )
	else:
            raise Exception ('Currently all blocks use ReLU')
        return conv_block


    
    def forward(self,x):
        if self.ds < 32:
            x = downsample(x, self.ds)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.dropout0(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.dropout1(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        #x = self.avgpool(x)
        x = avg_pool2d(x, kernel_size=x.size(2))
        x = x.view(x.size(0),-1)

        
        return x

