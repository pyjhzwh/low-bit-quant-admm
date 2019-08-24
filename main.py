#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import warnings
import time
import random
import accimage
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import *

import modelarchs
import admm

def test(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            #    images = images.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
            images, target = Variable(images.cuda()), Variable(target.cuda())

            #downsample the input since resnet18 is called by torchvision
            # but in resent 20 we have already write the downsample function
            if args.arch == 'resnet18':
                if args.ds != args.crop:
                    images = F.interpolate(images, args.ds, mode = 'bilinear')

            # compute output
            output = model(images)
            loss = criterion(output, target)
            #print("loss.size:",loss.size())
            #print("images.size(0)",images.size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.admm:
            if args.evaluate:
                weightsdistribute(model)

    return top1.avg

def mixtest(val_loader, model_big, model_small, epoch, args):
    model_big.eval()
    model_small.eval()

    confidence_small_record = torch.tensor([],dtype=torch.float32).cuda()#np.array([])
    confidence_big_record = torch.tensor([],dtype=torch.float32).cuda()#np.array([])
    pred_small_record = torch.tensor([], dtype=torch.int).cuda()#np.array([]).astype(int)
    pred_big_record = torch.tensor([], dtype=torch.int).cuda()#np.array([]).astype(int)
    target_record = torch.tensor([], dtype=torch.int).cuda()#np.array([]).astype(int)
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            images, target = Variable(images.cuda()), Variable(target.cuda())

            # Crop
            _, _, w, h = images.shape
            th, tw = args.crop, args.crop
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            #i (int): i in (i,j) i.e coordinates of the upper left corner.
            #j (int): j in (i,j) i.e coordinates of the upper left corner.
            #th (int): Height of the cropped image.
            #tw (int): Width of the cropped image.
            images_small = images[:,:,i:i+th,j:j+tw] 

            # ds
            # downsample the input since resnet18 is called by torchvision
            # but in resent 20 we have already write the downsample function
            if args.arch == 'resnet18':
                if args.ds != args.crop:
                    images_small = F.interpolate(images_small, args.ds, mode = 'bilinear')

            # compute output
            output_big = model_big(images)
            output_small = model_small(images_small)
            #loss = criterion(output, target)

            # confidence and pred of the output
            _, pred_big = output_big.topk(1, 1, True, True)
            _, pred_small = output_small.topk(1, 1, True, True)
            pred_big_record = torch.cat((pred_big_record, pred_big.type(torch.int).reshape(-1)), 0)
            pred_small_record = torch.cat((pred_small_record, pred_small.type(torch.int).reshape(-1)), 0)

            confidence_small = F.softmax(output_small.data, dim=1).max(1)[0]
            confidence_big = F.softmax(output_big.data, dim=1).max(1)[0]
            #print(confidence_small)
            confidence_small_record = torch.cat((confidence_small_record, confidence_small.type(torch.float).reshape(-1)), 0)
            confidence_big_record = torch.cat((confidence_big_record, confidence_big.type(torch.float).reshape(-1)), 0)

            target_record = torch.cat((target_record, target.type(torch.int)), 0)

        print('Threshold, Acc, big_compute_ratio, total_compute')

        sorted_confidence_small_record, _ = torch.sort(confidence_small_record,descending=True)
        #sorted_confidence_big_record, _ = torch.sort(confidence_big_record,descending=True)
        step = int(0.01* (len(confidence_small_record)))
        for index in range(0, len(confidence_small_record) , step):

            threshold = sorted_confidence_small_record[index]
            mask = (confidence_small_record <= threshold).type(torch.int)
            final_pred = pred_small_record * (1-mask) + pred_big_record * mask
            correct = (target_record == final_pred).float().sum(0)
            acc = correct.mul_(100.0 / len(confidence_small_record))
            big_compute_ratio =  mask.type(torch.float).sum(0) / len(confidence_small_record)
            total_compute = float(big_compute_ratio+ float(args.ds/224)**2)
            print('{:8.6f},{:4.2f},{:5.4f},{:5.4f}'.format(threshold,acc, big_compute_ratio, total_compute))

        small_acc = (target_record == pred_small_record).float().sum(0).mul_(100.0/ len(confidence_small_record))
        print('{:8.6f},{:4.2f},{:5.4f},{:5.4f}'.format(0 , small_acc , 0 , float((args.ds/224))**2))

        #print(target_record)
        print('relationship of confidence and acc')
        print('confidence , Acc1_small, Acc1_big')

        for i in range(10):
            thr_confidence = 1 - i*0.1
            mask_small = ((confidence_small_record <= thr_confidence) & (confidence_small_record > thr_confidence-0.1)).type(torch.int)
            mask_big = ((confidence_big_record <= thr_confidence) & (confidence_big_record > thr_confidence-0.1)).type(torch.int)
            acc_small = (pred_small_record * mask_small == target_record).float().sum(0) / mask_small.sum()
            acc_big = (pred_big_record * mask_big == target_record).float().sum(0) / mask_big.sum()

            print('{:.1f} - {:.1f},{:5.3f},{:5.3f}'.format(thr_confidence-0.1, thr_confidence,acc_small, acc_big))
            

        sys.stdout.flush()
        return acc



def train(train_loader,optimizer, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    model.train()


    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if args.gpu is not None:
            #images = images.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)
        images, target = Variable(images.cuda()), Variable(target.cuda())
        #downsample the input since resnet18 is called by torchvision
        # but in resent 20 we have already write the downsample function
        if args.arch == 'resnet18':
            if args.ds < 224:
                images = F.interpolate(images, args.ds, mode = 'bilinear')

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        # updata W gra, \partial_W L = \partial_W f + \rho (W-Z^K+U^K)
        if args.admm:
            admm.loss_grad()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 100 == 0:
            progress.display(i,optimizer)

    # update U,Z in W
    if args.admm:
        admm.update(epoch)
        admm.print_info(epoch)
        
        if epoch == args.epochs-1 :
            admm.apply_quantval()

    print('Finished Training')
    return

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__=='__main__':
    imagenet_datapath= '/data2/jiecaoyu/imagenet/imgs/'
    parser = argparse.ArgumentParser(description='PyTorch MNIST ResNet Example')
    parser.add_argument('--ds', type=int, default=32, 
            help = 'down sample size')
    parser.add_argument('--crop', type=int, default=32, 
            help = 'crop size')
    parser.add_argument('--no_cuda', default=False, 
            help = 'do not use cuda',action='store_true')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
            help='number of epochs to train (default: 450)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr_epochs', type=int, default=100, metavar='N',
            help='number of epochs to change lr (default: 100)')
    parser.add_argument('--pretrained', default=None, nargs='+',
            help='pretrained model ( for mixtest \
            the first pretrained model is the big one \
            and the sencond is the small net)')
    parser.add_argument('--resume', action='store_true',
                    default=False, help='resume start_epoch')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    default=False, help='evaluate model on validation set')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--arch', action='store', default='resnet20',
                        help='the CIFAR10 network structure: resnet20 | resnet18 | all_cnn_net')
    parser.add_argument('--dataset', action='store', default='cifar10',
            help='pretrained model: cifar10 | imagenet')
    parser.add_argument('-m', '--mix', dest='mix', action='store_true',
                    default=False, help='mix model of frontend and beckend, used with -e')
    parser.add_argument('--admm', action='store_true',
                    default=False, help='use admm to quantize weights')
    parser.add_argument('--admm-iter', default=10, type=int,
                    help='admm iter')
    parser.add_argument('--rho', default=1e-4, type=float,
                    help='admm rho parameter')
    parser.add_argument('--bits', default = [2,2,2,2,2,2,2,2,2], type = int,
                    nargs = '*', help = ' num of bits for each layer')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if not args.mix:
        testcrop = args.crop
    else:
        testcrop = 224

    if args.dataset == 'cifar10':
        # load cifa-10
        nclass = 10
        normalize = transforms.Normalize(
                mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=
                                            transforms.Compose([
                                                transforms.RandomCrop(args.crop,padding=2),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=16)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=
                                           transforms.Compose([
                                               transforms.RandomCrop(testcrop,padding=2),
                                               transforms.ToTensor(),
                                               normalize,
                                               ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=16)


    if args.dataset == 'imagenet':
        
        nclass = 100
        traindir = os.path.join(imagenet_datapath,'train')
        testdir = os.path.join(imagenet_datapath,'val')
        torchvision.set_image_backend('accimage')

        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trainset = torchvision.datasets.ImageFolder(root=traindir,transform=
                                            transforms.Compose([
                                                #transforms.Resize(256),
                                                #transforms.CenterCrop(args.crop),
                                                #transforms.RandomCrop(args.crop),
                                                transforms.RandomResizedCrop(args.crop, scale=(0.25, 1.0)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                normalize,
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=16)

        testset = torchvision.datasets.ImageFolder(root=testdir,transform=
                                           transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(testcrop),
                                               transforms.ToTensor(),
                                               normalize,
                                               ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=16)

    

    if args.arch == 'resnet20':
        if not args.mix:
            model = modelarchs.resnet20(nclass=nclass,ds=args.ds)
        else:
            model_big = modelarchs.resnet20(nclass=nclass,ds=args.ds)
            model_small = modelarchs.resnet20(nclass=nclass,ds=args.ds)
        

    elif args.arch == 'resnet18':
        pretrained = False if args.pretrained is not None else True
        if not args.mix:
            model = torchvision.models.resnet18(pretrained = pretrained)
            bestacc = 0
        else:
            model_big = torchvision.models.resnet18(pretrained = pretrained)
            model_small = torchvision.models.resnet18(pretrained = pretrained)
            bestacc = 0

    elif args.arch == 'all_cnn_c':
        if not args.mix:
            model = modelarchs.all_cnn_c()

    #print("--------model state dict--------")
    #for key, _ in model.named_parameters():
    #    print(key)
    criterion = nn.CrossEntropyLoss().cuda()
    if not  args.mix:
        optimizer = optim.SGD(model.parameters(), 
                lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)

    if not args.pretrained:
        bestacc = 0
    elif not args.mix:
        pretrained_model = torch.load(args.pretrained[0])
        #print('bestacc',bestacc)
        if args.resume: # resume from previous training, otherwise just load parameters
            args.start_epoch = pretrained_model['epoch']
            bestacc = pretrained_model['acc'].item()
        else:
            bestacc = 0
        load_state(model, pretrained_model['state_dict'])
        optimizer.load_state_dict(pretrained_model['optimizer'])
        to_cuda_optimizer(optimizer)
    else:
        pretrained_model_big = torch.load(args.pretrained[0])
        bestacc = pretrained_model_big['acc']
        args.start_epoch = pretrained_model_big['epoch']
        load_state(model_big, pretrained_model_big['state_dict'])

        pretrained_model_small = torch.load(args.pretrained[1])
        bestacc = pretrained_model_small['acc']
        args.start_epoch = pretrained_model_small['epoch']
        load_state(model_small, pretrained_model_small['state_dict'])

    if args.cuda:
        if not args.mix:
            model.cuda()
            model = nn.DataParallel(model, 
                    device_ids=range(torch.cuda.device_count()))
        #model = nn.DataParallel(model, device_ids=args.gpu)
        else:
            model_big.cuda()
            model_big = nn.DataParallel(model_big, 
                    device_ids=range(torch.cuda.device_count()))
            model_small.cuda()
            model_small = nn.DataParallel(model_small, 
                    device_ids=range(torch.cuda.device_count()))


    if not args.mix:
        print(model)
    else:
        print(model_big)

    # admm

    if args.admm:
        #if args.arch == 'all_cnn_c':
            #bits = [1,2,2,2,2,2,2,2,2] 
        admm = admm.admm_op(model,b=args.bits,admm_iter=args.admm_iter)


    ''' evaluate model accuracy and loss only '''
    if args.evaluate:
        if not args.mix:
            test(testloader, model, args.start_epoch, args)
            exit()
        else:
            #test(testloader, model_small, args.start_epoch, args)
            mixtest(testloader, model_big, model_small, args.start_epoch, args)
            exit()

    ''' train model '''

    for epoch in range(args.start_epoch,args.epochs):
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch, args)
        train(trainloader,optimizer, model, epoch, args)
        acc = test(testloader, model, epoch, args)
        if (acc > bestacc):
            bestacc = acc
            save_state(model,bestacc,epoch,args, optimizer, True)
        else:
            save_state(model,bestacc,epoch,args,optimizer, False)
        print('best acc so far:{:4.2f}'.format(bestacc))

    if args.admm:
        # save the last quantized value
        save_state(model,acc,epoch,args, optimizer, True)
        weightsdistribute(model)
        total_bit = 0
        total_param = 0
        i = 0
        for key, value in model.named_parameters():
            if '.0.weight' in key:
                total_param = total_param + value.numel()
                total_bit = total_bit + value.numel() * args.bits[i]
                i = i + 1

        print('aver bits: {:10d} / {:5d} = {:5.3f}'.format(total_bit, total_param, total_bit / total_param))

