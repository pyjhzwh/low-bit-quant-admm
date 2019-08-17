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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample

        self.stride = stride

        return

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        #print('residual size:',residual.size())

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        #print('out size:',out.size())
        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, nclass=10, ds=False):
        super(ResNet,self).__init__()
        self.nclass = nclass
        self.ds = ds
        self.inplanes = 16
        self.conv1 = conv3x3(3,self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block,16,layers[0],stride=1)
        self.layer2 = self._make_layer(block,32,layers[1],stride=2)
        self.layer3 = self._make_layer(block,64,layers[2],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64* block.expansion, nclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.normal_(m.weight)
                #nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)



    def _make_layer(self, block, planes, blocks, stride =1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print("downsample, stride = ",stride)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        if self.ds < 32:
            x = downsample(x, self.ds)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

class CatResNet(nn.Module):
    def __init__(self, block, layers, nclass=10, ds=False):
        super(ResNet,self).__init__()
        self.nclass = nclass
        self.ds = ds
        self.inplanes = 16
        self.net0 = ResNet(BasicBlock, layers, nclass=nclass, ds=ds)
        self.net1 = ResNet(BasicBlock, layers, nclass=nclass, ds=ds)
        self.net2 = ResNet(BasicBlock, layers, nclass=nclass, ds=ds)
        self.p0 = torch.tensor([1/3],requires_grad=True)
        self.p1 = torch.tensor([1/3],requires_grad=True)
        self.p2 = torch.tensor([1/3],requires_grad=True)


    def forward(self,x):
        x = x.view()
        x0 = self.net0(x[0])
        x1 = self.net1(x[1])
        x2 = self.net2(x[2])
        out = tensor.add(
                tensor.mul(x0,self.p0),
                tensor.mul(x1,self.p1), 
                tensor.mul(x2,self.p2))
        return out

        '''
        for net in nnet:
            if self.ds < 32:
                x[net] = downsample(x[net], self.ds)

            x[net] = self.conv1(x[net])
            x[net] = self.bn1(x[net])
            x[net] = self.relu(x[net])

            x[net] = self.layer1(x[net])
            x[net] = self.layer2(x[net])
            x[net] = self.layer3(x[net])

            x[net] = self.avgpool(x[net])
            x[net] = x[net].view(x[net].size(0),-1)
            x[net] = self.fc(x[net])
        '''

def resnet20(nclass=10, ds=32):
    return ResNet(BasicBlock, [3,3,3], nclass=nclass, ds=ds)

#def cat_resnet20(nclass=10, ds=8, nnet=3)
