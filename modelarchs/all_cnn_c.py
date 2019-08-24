#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys

def ConvBlock(in_planes, out_planes, kernel_size=-1, padding=-1, stride=-1, relu=False):
    if relu:
        conv_block = nn.Sequential(
                nn.Conv2d(in_planes, out_planes,
                    kernel_size=kernel_size, padding=padding, stride=stride,
                    bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.01),
                nn.ReLU(inplace=True)
                )
    else:
        raise Exception ('Currently all blocks use ReLU')
    return conv_block

class all_cnn_c(nn.Module):
    def __init__(self, ds=False):
        super(all_cnn_c, self).__init__()

        self.ds = ds

        self.conv0 = ConvBlock(3, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv1 = ConvBlock(96, 96, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv2 = ConvBlock(96, 96, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout0 = nn.Dropout(0.5)

        self.conv3 = ConvBlock(96, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv4 = ConvBlock(192, 192, kernel_size=3, padding=1, stride=1, relu=True)
        self.conv5 = ConvBlock(192, 192, kernel_size=3, padding=1, stride=2, relu=True)

        self.dropout1 = nn.Dropout(0.5)

        self.conv6 = ConvBlock(192, 192, kernel_size=3, padding=0, stride=1, relu=True)
        self.conv7 = ConvBlock(192, 192, kernel_size=1, padding=0, stride=1, relu=True)
        self.conv8 = ConvBlock(192, 10, kernel_size=1, padding=0, stride=1, relu=True)

        # self.avg_pool = nn.AvgPool2d(kernel_size=6)

        self.__initialize__()

        if ds:
            self.conv6[0].padding=(1,1)

        return

    def __initialize__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return

    def forward(self, x):
        #if self.ds:
            #x = F.avg_pool2d(x, kernel_size=2, stride=2)
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)

        out = self.dropout0(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.dropout1(out)

        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)

        # out = self.avg_pool(out)
        out = F.avg_pool2d(out, kernel_size=out.size(2))
        out = out.view(out.size(0), -1)
        return out
