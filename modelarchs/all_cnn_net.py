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



class all_cnn_c(nn.Module):
    def __init__(self, nclass=10, ds=False):
        super(all_cnn_c,self).__init__()
        self.nclass = nclass
        self.ds = ds

        self.dropout0 = nn.Dropout(p=0.2)

        self.conv0 = self._conv_block(3, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv1 = self._conv_block(96, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv2 = self._conv_block(96, 96, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout1 = nn.Dropout(p=0.5)

        self.conv3 = self._conv_block(96, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv4 = self._conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv5 = self._conv_block(192, 192, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout2 = nn.Dropout(p=0.5)

        self.conv6 = self._conv_block(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv7 = self._conv_block(192, 192, kernel_size=1, padding=0, stride=1, relu=True)
        self.conv8 = self._conv_block(192, 10, kernel_size=1, padding=0, stride=1, relu=True)

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


    def _conv_block(self, in_planes, out_planes, kernel_size=1, padding=1,
            stride = 1, relu = True):
        if relu:
            conv_block = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes, momentum=0.01),
                    nn.ReLU(inplace = True)
                    )
        else:
            raise Exception ('Currently all blocks use ReLU')
        return conv_block


    
    def forward(self,x):
        if self.ds < 32:
            x = downsample(x, self.ds)

        x = self.dropout0(x)
        
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        #x = self.avgpool(x)
        x = F.avg_pool2d(x, kernel_size=x.size(2))
        x = x.view(x.size(0),-1)

        
        return x

