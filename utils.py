#!/usr/bin/env python3

import argparse
import os
import warnings
import time
import random
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import modelarchs

def save_state(model, best_acc, epoch, args,optimizer, isbest):
    dirpath = 'saved_models/'
    suffix = '.ckp_origin.pth.tar'
    state = {
            'acc': best_acc,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'isbest': isbest,
            }
    if args.dataset == 'imagenet':
        orgsize = 224
    elif args.dataset == 'cifar10':
        orgsize = 32
    #if args.crop < orgsize:
    if not args.admm:
        filename = str(args.arch)+'.'+str(args.ds)+'.'+str(args.crop)+suffix
    else:
        filename = 'admm.'+str(args.arch)+'.'+str(args.ds)+'.'+str(args.crop)+suffix
    #else:
        #filename = str(args.arch)+'.'+str(args.ds)+'.'+str(args.crop)+'.hcha'.suffix

        #torch.save(state,'saved_models/{}.{}.{}.ckp_origin.pth.tar'.format(args.arch,args.ds,args.crop))
    #else: 
        #filename = str(args.arch)+'.'+str(args.ds)+suffix
        #torch.save(state,'saved_models/{}.{}.ckp_origin.pth.tar'.format(args.arch,args.ds))
    torch.save(state,dirpath+filename)
    if isbest:
        shutil.copyfile(dirpath+filename, dirpath+'best.'+filename)
    
    #torch.save(state,'saved_models/{}.{}.{}.ckp_origin.pth.tar'.format(args.arch,args.ds,args.crop))
    return

def load_state(model, state_dict):
    cur_state_dict = model.state_dict()
    state_dict_keys = state_dict.keys()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])

    
    #model.load_state_dict(state_dict)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch,optimizer=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if optimizer is not None:
            entries += ['lr: {:.1e}'.format(optimizer.param_groups[0]['lr'])]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every lr-epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def to_cuda_optimizer(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

def weightsdistribute(model):
    for key, value in model.named_parameters():
        unique, count = torch.unique(value, sorted=True, return_counts= True)
        print(unique,":",count)

