# # -*- coding: utf-8 -*-
#
# import time
# import os
# import math
# import argparse
# from glob import glob
# from collections import OrderedDict
# import random
# import warnings
# import datetime
#
# import numpy as np
# from tqdm import tqdm
#
# from sklearn.model_selection import train_test_split
# import joblib
# from skimage.io import imread
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.utils.data import DataLoader
# import torch.backends.cudnn as cudnn
# from ranger21 import Ranger21
# import torchvision
# from torchvision import datasets, models, transforms
#
# from dataset.dataset import Dataset
#
# from utilspag.metrics import dice_coef, batch_iou, mean_iou, iou_score
# import utilspag.losses as losses
# from utilspag.utils import str2bool, count_params
# import pandas as pd
# from net import UNet, ResNet, SegNet, KiuNet, kiunetorg, transunet, transright, volo
#
# torch.backends.cudnn.enabled = True
# arch_names = list(UNet.__dict__.keys())
# loss_names = list(losses.__dict__.keys())
# loss_names.append('BCEWithLogitsLoss')
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--name', default=None,
#                         help='model name: (default: arch+timestamp)')
#     parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
#                         choices=arch_names,
#                         help='model architecture: ' +
#                              ' | '.join(arch_names) +
#                              ' (default: NestedUNet)')
#     """和常规的深度学习机制相比，深度监督学习不仅在网络的最后输出结果out，同时在网络的中间特征图，经过反卷积和上采样操作，
#     得到和out尺寸一致的输出out_m，然后结合out-m和out，共同训练网络。在GoogleNet中用到了这种学习机制。"""
#     parser.add_argument('--deepsupervision', default=True, type=str2bool)
#     parser.add_argument('--dataset', default="LITS",
#                         help='dataset name')
#     parser.add_argument('--input-channels', default=1, type=int,
#                         help='input channels')
#     parser.add_argument('--image-ext', default='npy',
#                         help='image file extension')
#     parser.add_argument('--mask-ext', default='npy',
#                         help='mask file extension')
#     parser.add_argument('--aug', default=False, type=str2bool)
#     parser.add_argument('--loss', default='BCEWithLogitsLoss',
#                         choices=loss_names,
#                         help='loss: ' +
#                              ' | '.join(loss_names) +
#                              ' (default: BCEDiceLoss)')
#     parser.add_argument('--epochs', default=20, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('--early-stop', default=100, type=int,
#                         metavar='N', help='early stopping (default: 20)')
#     parser.add_argument('-b', '--batch-size', default=4, type=int,
#                         metavar='N', help='mini-batch size (default: 4)')
#     parser.add_argument('--optimizer', default='Ranger21',
#                         choices=['Ranger21', 'SGD'],
#                         help='loss: ' +
#                              ' | '.join(['Ranger21', 'SGD']) +
#                              ' (default: Ranger21)')
#     parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                         metavar='LR', help='initial learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float,
#                         help='momentum')
#     parser.add_argument('--weight-decay', default=1e-4, type=float,
#                         help='weight decay')
#     parser.add_argument('--nesterov', default=False, type=str2bool,
#                         help='nesterov')
#     """把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中"""
#     args = parser.parse_args()
#
#     return args
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     """在train函数中采用自定义的AverageMeter类来管理一些变量的更新"""
#
#     def __init__(self):
#         self.reset()
#
#     """在初始化的时候就调用的重置方法reset"""
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     """当调用该类对象的update方法的时候就会进行变量更新，当要读取某个变量的时候，可以通过对象.属性的方式来读取"""
#     """val代表本次iterion的预测值"""
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         """self.avg代表平均值"""
#         self.avg = self.sum / self.count
#
#
# """args 数据加载器 模型 损失函数 优化器"""
#
#
# def train(args, train_loader, model, criterion, optimizer, epoch, alpha, scheduler=None):
#     losses = AverageMeter()
#     ious = AverageMeter()
#     dices_1s = AverageMeter()
#     dices_2s = AverageMeter()
#     model.train()
#
#     for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
#
#         input = input.cuda()
#         target = target.cuda()
#
#         # compute output
#         if args.deepsupervision:
#             outputs = model(input)
#
#             loss0 = criterion(outputs[0], target)
#             loss1 = criterion(outputs[1], target)
#             loss2 = criterion(outputs[2], target)
#             loss3 = criterion(outputs[3], target)
#
#             loss = loss3 + alpha * (loss0 + loss1 + loss2)
#             iou = iou_score(outputs[-1], target)
#             dice_1 = dice_coef(outputs[-1], target)[0]
#             dice_2 = dice_coef(outputs[-1], target)[1]
#         else:
#             output = model(input)
#
#             loss = criterion(output, target)
#             """分割指标"""
#             iou = iou_score(output, target)
#             """"评价指标"""
#             dice_1 = dice_coef(output, target)[0]
#             dice_2 = dice_coef(output, target)[1]
#
#         losses.update(loss.item(), input.size(0))
#         ious.update(iou, input.size(0))
#         dices_1s.update(torch.tensor(dice_1), input.size(0))
#         dices_2s.update(torch.tensor(dice_2), input.size(0))
#
#         # compute gradient and do optimizing step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     log = OrderedDict([
#         ('loss', losses.avg),
#         ('iou', ious.avg),
#         ('dice_1', dices_1s.avg),
#         ('dice_2', dices_2s.avg)
#     ])
#
#     return log
#
#
# def validate(args, val_loader, model, criterion, alpha):
#     losses = AverageMeter()
#     ious = AverageMeter()
#     dices_1s = AverageMeter()
#     dices_2s = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
#             input = input.cuda()
#             target = target.cuda()
#
#             # compute output
#             if args.deepsupervision:
#                 output = model(input)
#                 loss = criterion(output, target)
#                 iou = iou_score(output[-1], target)
#                 dice_1 = dice_coef(output, target)[0]
#                 dice_2 = dice_coef(output, target)[1]
#             else:
#                 output = model(input)
#                 loss = criterion(output, target)
#                 iou = iou_score(output, target)
#                 dice_1 = dice_coef(output, target)[0]
#                 dice_2 = dice_coef(output, target)[1]
#
#             losses.update(loss.item(), input.size(0))
#             ious.update(iou, input.size(0))
#             dices_1s.update(torch.tensor(dice_1), input.size(0))
#             dices_2s.update(torch.tensor(dice_2), input.size(0))
#
#     log = OrderedDict([
#         ('loss', losses.avg),
#         ('iou', ious.avg),
#         ('dice_1', dices_1s.avg),
#         ('dice_2', dices_2s.avg)
#     ])
#
#     return log
#
#
# def init(module):
#     if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
#         nn.init.kaiming_normal_(module.weight.data, 0.5)
#         nn.init.constant_(module.bias.data, 0)
#
#
# def main():
#     args = parse_args()
#     # args.dataset = "datasets"
#
#     if args.name is None:
#         if args.deepsupervision:
#             args.name = '%s_%s_lym' % (args.dataset, args.arch)
#         else:
#             args.name = '%s_%s_lym' % (args.dataset, args.arch)
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     if not os.path.exists('models/{}/{}'.format(args.name, timestamp)):
#         os.makedirs('models/{}/{}'.format(args.name, timestamp))
#
#     """打印parse_args参数"""
#     print('Config -----')
#     for arg in vars(args):
#         print('%s: %s' % (arg, getattr(args, arg)))
#     print('------------')
#
#     """"将参数写入.txt文档"""
#     with open('models/{}/{}/args.txt'.format(args.name, timestamp), 'w') as f:
#         for arg in vars(args):
#             print('%s: %s' % (arg, getattr(args, arg)), file=f)
#     """模型保存"""
#     joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name, timestamp))
#
#     # define loss function (criterion)
#     if args.loss == 'BCEDiceLoss':
#         print("我不会执行")
#         criterion = nn.BCEWithLogitsLoss().cuda()
#     else:
#         print("我被执行了")
#         criterion = losses.BCEDiceLoss().cuda()
#         # criterion = losses.LogNLLLoss().cuda()
#
#     cudnn.benchmark = True
#
#     # Data loading code
#     train_img_paths = glob('./data/train_image/*')
#     train_mask_paths = glob('./data/train_mask/*')
#     val_img_paths = glob('./data/val_image/*')
#     val_mask_paths = glob('./data/val_mask/*')
#     print("train_num:%s" % str(len(train_img_paths)))
#     print("val_num:%s" % str(len(val_img_paths)))
#
#     # create model
#     # 查看使用的哪个模型
#     print("=> creating model %s" % args.arch)
#     # model = UNet.UNet3d(in_channels=1, n_classes=2, n_channels=32)
#     model = ResNet.ResUNet(1, 2, True)
#     model.apply(init)
#
#     # model = SegNet.SegNet(True)
#     # model = KiuNet.kiunet_min(True)
#     # model = kiunetorg.kiunet_org(True)
#     # model = transright.UNeTR()
#     # model = volo.UNETR()
#     # model = transunet.UNeTR()
#     model = torch.nn.DataParallel(model).cuda()
#     # model._initialize_weights()
#     # model.load_state_dict(torch.load('model.pth'))
#
#     print(count_params(model))
#
#
#     train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
#     val_dataset = Dataset(args, val_img_paths, val_mask_paths)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         drop_last=True)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         pin_memory=True,
#         drop_last=False)
#
#     log = pd.DataFrame(index=[], columns=[
#         'epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'
#     ])
#     if args.optimizer == 'Ranger21':
#         optimizer = Ranger21(model.parameters(), lr=args.lr, num_epochs=args.epochs, num_batches_per_epoch=len(train_loader))
#     elif args.optimizer == 'SGD':
#         optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
#                               momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
#     best_loss = 100
#     # best_iou = 0
#     trigger = 0
#     first_time = time.time()
#
#     for epoch in range(args.epochs):
#
#         print('Epoch [%d/%d]' % (epoch, args.epochs))
#         alpha = 0.33
#         if epoch % 40 is 0 and epoch is not 0:
#             alpha *= 0.8
#         # train for one epoch
#         train_log = train(args, train_loader, model, criterion, optimizer, epoch, alpha)
#         # evaluate on validation set
#         val_log = validate(args, val_loader, model, criterion, alpha)
#
#         print(
#             'loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - '
#             'val_dice_2 %.4f '
#             % (train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2'], val_log['loss'],
#                val_log['iou'], val_log['dice_1'], val_log['dice_2']))
#
#         end_time = time.time()
#         print("time:", (end_time - first_time) / 60)
#
#         """tmp为输出的参数值"""
#         tmp = pd.Series([
#             epoch,
#             args.lr,
#             train_log['loss'],
#             train_log['iou'],
#             train_log['dice_1'],
#             train_log['dice_2'],
#             val_log['loss'],
#             val_log['iou'],
#             val_log['dice_1'],
#             val_log['dice_2'],
#         ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'])
#
#         log = log.append(tmp, ignore_index=True)
#         log.to_csv('models/{}/{}/log.csv'.format(args.name, timestamp), index=False)
#
#         trigger += 1
#
#         val_loss = val_log['loss']
#         if val_loss < best_loss:
#             torch.save(model.state_dict(),
#                        'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format(args.name, timestamp, epoch,
#                                                                              val_log['dice_1'], val_log['dice_2']))
#             best_loss = val_loss
#             print("=> saved best model")
#             trigger = 0
#
#         checkpoint = {
#             "net": model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             "epoch": epoch
#         }
#         if epoch % 5 == 0:
#             torch.save(checkpoint, './models/check/ckpt_best_%s.pth' % (str(epoch)))
#
#         # early stopping
#         if not args.early_stop is None:
#             if trigger >= args.early_stop:
#                 print("=> early stopping")
#                 break
#
#         torch.cuda.empty_cache()
#
#
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#     main()
# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from ranger21 import Ranger21

from dataset.dataset import Dataset

from utilspag.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilspag.losses as losses
from utilspag.utils import str2bool, count_params
import pandas as pd
from net import UNet, ResNet, SegNet, KiuNet, kiunetorg, transunet, transright, volo, kTUnet

torch.backends.cudnn.enabled = True
arch_names = list(UNet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=arch_names,
                        help='model architecture: ' +
                             ' | '.join(arch_names) +
                             ' (default: NestedUNet)')
    """和常规的深度学习机制相比，深度监督学习不仅在网络的最后输出结果out，同时在网络的中间特征图，经过反卷积和上采样操作，
    得到和out尺寸一致的输出out_m，然后结合out-m和out，共同训练网络。在GoogleNet中用到了这种学习机制。"""
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="LITS",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEWithLogitsLoss',
                        choices=loss_names,
                        help='loss: ' +
                             ' | '.join(loss_names) +
                             ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=30, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--optimizer', default='Ranger',
                        choices=['Ranger', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Ranger', 'SGD']) +
                             ' (default: Ranger)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    """把parser中设置的所有"add_argument"给返回到args子类实例当中， 那么parser中增加的属性内容都会在args实例中"""
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    """在train函数中采用自定义的AverageMeter类来管理一些变量的更新"""

    def __init__(self):
        self.reset()

    """在初始化的时候就调用的重置方法reset"""

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    """当调用该类对象的update方法的时候就会进行变量更新，当要读取某个变量的时候，可以通过对象.属性的方式来读取"""
    """val代表本次iterion的预测值"""

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        """self.avg代表平均值"""
        self.avg = self.sum / self.count


"""args 数据加载器 模型 损失函数 优化器"""


def train(args, train_loader, model, criterion, optimizer, epoch, alpha, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        input = input.cuda()
        target = target.cuda()

        # compute output
        if args.deepsupervision:
            outputs = model(input)

            loss0 = criterion(outputs[0], target)
            loss1 = criterion(outputs[1], target)
            loss2 = criterion(outputs[2], target)
            loss3 = criterion(outputs[3], target)

            loss = loss3 + alpha * (loss0 + loss1 + loss2)
            iou = iou_score(outputs[-1], target)
            dice_1 = dice_coef(outputs[-1], target)[0]
            dice_2 = dice_coef(outputs[-1], target)[1]
        else:
            output = model(input)

            loss = criterion(output, target)
            """分割指标"""
            iou = iou_score(output, target)
            """"评价指标"""
            dice_1 = dice_coef(output, target)[0]
            dice_2 = dice_coef(output, target)[1]

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))
        dices_2s.update(torch.tensor(dice_2), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output[-1], target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def main():
    args = parse_args()
    # args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' % (args.dataset, args.arch)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name, timestamp)):
        os.makedirs('models/{}/{}'.format(args.name, timestamp))

    """打印parse_args参数"""
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    """"将参数写入.txt文档"""
    with open('models/{}/{}/args.txt'.format(args.name, timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)
    """模型保存"""
    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name, timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEDiceLoss':
        print("我不会执行")
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        print("我被执行了")
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    train_img_paths = glob('./data/train_image/*')
    train_mask_paths = glob('./data/train_mask/*')
    val_img_paths = glob('./data/val_image/*')
    val_mask_paths = glob('./data/val_mask/*')
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    # 查看使用的哪个模型
    print("=> creating model %s" % args.arch)
    # model = UNet.UNet3d(in_channels=1, n_classes=2, n_channels=32)
    # model = ResNet.ResUNet(1, 2, True)
    # model = SegNet.SegNet(True)
    # model = KiuNet.kiunet_min(True)
    model = kTUnet.UNETR()
    # model = transright.UNeTR()
    # model = volo.UNETR()
    # model = kTUnet.UNETR()
    # model = transunet.UNETR()
    # model.apply(init)

    model = torch.nn.DataParallel(model).cuda()
    # model._initialize_weights()
    # model.load_state_dict(torch.load('model.pth'))

    print(count_params(model))

    # if args.optimizer == 'Ranger':
    #     optimizer = Ranger21(model.parameters(), lr=args.lr)
    #     # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # elif args.optimizer == 'SGD':
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #                           momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'
    ])

    if args.optimizer == 'Ranger':
        optimizer = Ranger21(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                             num_epochs=args.epochs, num_batches_per_epoch=len(train_loader))
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


    best_loss = 100
    # best_iou = 0
    trigger = 0
    first_time = time.time()
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        alpha = 0.33
        if epoch % 40 == 0 and epoch != 0:
            alpha *= 0.8
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, alpha)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print(
            'loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - '
            'val_dice_2 %.4f '
            % (train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2'], val_log['loss'],
               val_log['iou'], val_log['dice_1'], val_log['dice_2']))

        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        """tmp为输出的参数值"""
        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],
            train_log['dice_2'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],
            val_log['dice_2'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name, timestamp), index=False)

        trigger += 1

        val_loss = val_log['loss']
        if val_loss < best_loss:
            torch.save(model.state_dict(),
                       'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format(args.name, timestamp, epoch,
                                                                             val_log['dice_1'], val_log['dice_2']))
            best_loss = val_loss
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()
