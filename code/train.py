from __future__ import print_function
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES']= '1'
# from dice_loss import TverskyLoss
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations.core.composition import Compose, OneOf
import albumentations as albu

import archs
import losses
from dataset import Dataset
from metrics import *
import utils
from utils import AverageMeter, str2bool, gram
from resnet50 import Resnet50


import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
#
# from LoadData import Dataset, loader, Dataset_val
import itertools


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

STYLE_WEIGHT = 1e0
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7

torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=224, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='liver2',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')


    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')


    # parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
  #  epoch  parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
  #   parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    #  weight_decay parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
    # parser.add_argument('--cuda', action='store_true', default=True, help='using GPU or not')
    parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
    # parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')

    parser.add_argument('--cuda', action='store_true', default=True, help='using GPU or not')
    parser.add_argument('--outpath', default='outputGANliver3', help='folder to output images and model checkpoints')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize
    return  dice_total

class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                                   0.3 * (pred * (1 - target)).sum(dim=1).sum(
                    dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 2)



def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}


    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
       

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),      
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
               
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
               

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
           


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),      
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                      ])


config = vars(parse_args())

cuda = config['cuda']
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
#
# torch.manual_seed(config.seed)
# if cuda:
#     torch.cuda.manual_seed(config.seed)

cudnn.benchmark = True
print('===> Building model')

print("=> creating D model")
# NetD = Resnet50().cuda
NetD = Resnet50().to(device)
# NetS.apply(weights_init)
#print(NetD)
print("=> created D model Resnet50")
print("=> creating G model")


if config['name'] is None:
    if config['deep_supervision']:
        config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
    else:
        config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
os.makedirs('models/%s' % config['name'], exist_ok=True)

print('-' * 20)
for key in config:
    print('%s: %s' % (key, config[key]))
print('-' * 20)

with open('models/%s/config.yml' % config['name'], 'w') as f:
    yaml.dump(config, f)

# define loss function (criterion)
if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = losses.__dict__[config['loss']]().cuda()

cudnn.benchmark = True

NetG = archs.__dict__[config['arch']](config['num_classes'],
                                       config['input_channels'],
                                       config['deep_supervision'])
# NetG = NetG.cuda()
NetG = NetG.to(device)
#print(NetG)
print("=> created D model")
if cuda:

    NetG = NetG.cuda()
    NetD = NetD.cuda()

lr = config['lr']

optimizerD = optim.Adam(NetD.parameters(), lr=lr, betas=(config['beta1'], 0.999))

params = filter(lambda p: p.requires_grad,  itertools.chain(NetD.parameters(),NetG.parameters()))
if config['optimizer'] == 'Adam':
    optimizerG = optim.Adam(
        params, lr=config['lr'], weight_decay=config['weight_decay'])
elif config['optimizer'] == 'SGD':
    optimizerG = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                          nesterov=config['nesterov'], weight_decay=config['weight_decay'])
else:
    raise NotImplementedError

if config['scheduler'] == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizerG, T_max=config['epochs'], eta_min=config['min_lr'])
elif config['scheduler'] == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizerG, factor=config['factor'], patience=config['patience'],
                                               verbose=1, min_lr=config['min_lr'])
elif config['scheduler'] == 'MultiStepLR':
    scheduler = lr_scheduler.MultiStepLR(optimizerG, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
elif config['scheduler'] == 'ConstantLR':
    scheduler = None
else:
    raise NotImplementedError

# Data loading code
img_ids = glob(os.path.join('inputs', config['dataset'], 'training/images', '*' + config['img_ext']))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)


train_transform = Compose([
    albu.Rotate(limit=90, p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.OneOf([
        albu.HueSaturationValue(p=1),
        #albu.RandomBrightness(),
        #albu.RandomContrast(),
        albu.RandomBrightnessContrast(p=1),
    ], p=1),
    albu.Resize(config['input_h'], config['input_w']),  
    albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  
])

   
val_transform = Compose([
    # transforms.Resize(config['input_h'], config['input_w']),
    albu.Resize(config['input_h'], config['input_w']),
    albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
    
train_dataset = Dataset(
    img_ids=train_img_ids,
    img_dir=os.path.join('inputs', config['dataset'], 'training/images'),
    mask_dir=os.path.join('inputs', config['dataset'], 'training/masks'),
    img_ext=config['img_ext'],
    mask_ext=config['mask_ext'],
    num_classes=config['num_classes'],
    transform=train_transform)
val_dataset = Dataset(
    img_ids=val_img_ids,
    img_dir=os.path.join('inputs', config['dataset'], 'training/images'),
    mask_dir=os.path.join('inputs', config['dataset'], 'training/masks'),
    img_ext=config['img_ext'],
    mask_ext=config['mask_ext'],
    num_classes=config['num_classes'],
    transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False)

log = OrderedDict([
    ('epoch', []),
    # ('lr', []),
    # ('loss', []),
    ('train_iou', []),
    ('train_dice', []),
    # ('val_loss', []),
    ('val_iou', []),
    ('val_dice', []),
    ('G_Loss', []),
    # ('D_Loss', [])

])
torch.cuda.empty_cache()
max_iou = 0
best_dice = 0
best_sen = 0
best_pre = 0
best_F1 = 0
loss_mse = torch.nn.MSELoss()
NetG.train()
for epoch in range(config['epochs']):

    aggregate_style_loss = 0.0
    aggregate_content_loss = 0.0
    aggregate_tv_loss = 0.0

    for i, data in enumerate(train_loader , 1):
        # train D
        NetD.zero_grad()
        input, label = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            target = label.cuda()

        target = target.type(torch.FloatTensor)
        target = target.cuda()
        output = NetG(input).to(device)
     
        output_masked = input.clone()
        input_mask = input.clone()
    
        # detach G from the network

        for d in range(3):
            output_masked[:,d,:,:] = input_mask[:,d,:,:] * (output.squeeze(1))
        if cuda:
            output_masked = output_masked.cuda()

        result = NetD(output_masked)
        target_masked = input.clone()
        for d in range(3):
            target_masked[:,d,:,:] = input_mask[:,d,:,:] * (target.squeeze(1))
        if cuda:
            target_masked = target_masked.cuda()
        target_D = NetD(target_masked)

        y_hat=output
        y_c_features = target_D
        y_hat_features = result

        
        target_D_ = [fmap for fmap in y_c_features]

        y_hat_ = [fmap for fmap in y_hat_features]

        style_loss = 0.0
        for j in range(4):
            style_loss += loss_mse(y_hat_[j], target_D_[j])
        style_loss = STYLE_WEIGHT * style_loss
        aggregate_style_loss += style_loss.item()
        recon = y_c_features[1]
        recon_hat = y_hat_features[1]
        content_loss = CONTENT_WEIGHT * loss_mse(recon_hat, recon)
        aggregate_content_loss += content_loss.item()

  
   
        NetG.zero_grad()
        output = NetG(input)
  
        for d in range(3):
            output_masked[:,d,:,:] = input_mask[:,d,:,:] * (output.squeeze(1))
        if cuda:
            output_masked = output_masked.cuda()
        result = NetD(output_masked)
        result=torch.cat((torch.tensor(result[0]).view(config['batch_size'], -1), torch.tensor(result[1]).view(config['batch_size'], -1),
                         torch.tensor(result[2]).view(config['batch_size'], -1), torch.tensor(result[3]).view(config['batch_size'], -1)), 1)
        for d in range(3):
            target_masked[:,d,:,:] = input_mask[:,d,:,:]* (target.squeeze(1))
        if cuda:
            target_masked = target_masked.cuda()
        target_G = NetD(target_masked)
        target_G = torch.cat(
            (torch.tensor(target_G[0]).view(config['batch_size'], -1), torch.tensor(target_G[1]).view(config['batch_size'], -1),
             torch.tensor(target_G[2]).view(config['batch_size'], -1), torch.tensor(target_G[3]).view(config['batch_size'], -1)), 1)
        loss_dice = criterion(output,target)
        loss_G = torch.mean(torch.abs(result - target_G))
        TverskyLoss1=TverskyLoss().cuda()
        Tversky_Loss=TverskyLoss1(output,target)
        loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice + Tversky_Loss
        loss_G_joint.backward()
        optimizerG.step()

    print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(train_loader), 1 - loss_dice.data))
    print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(train_loader), loss_G.data))
  
    NetG.eval()
    IoUs, dices, sens, pres, F1s = [], [], [], [], [] 
    for i, data in enumerate(val_loader, 1):
        input, gt = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            gt = gt.cuda()
        pred = NetG(input)
        pred = pred.type(torch.LongTensor)
        for x in range(input.size()[0]):
            IoU = iou_score(pred[x], gt[x])
            dice = dice_coef(pred[x], gt[x])
            sen = get_sensitivity(pred[x], gt[x])
            pre = get_precision(pred[x], gt[x])
            F1 = get_F1(pred[x], gt[x])
            IoUs.append(IoU)
            dices.append(dice)
            sens.append(sen)
            pres.append(pre)
            F1s.append(F1)

    train_IoUs, train_dices = [], []
    for i, data in enumerate(train_loader, 1):
        input, gt = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            gt = gt.cuda()
        pred = NetG(input)
        pred = pred.type(torch.LongTensor)
        for x in range(input.size()[0]):
            train_IoU=iou_score(pred[x],gt[x])
            train_dice=dice_coef(pred[x], gt[x])
            train_IoUs.append(IoU)
            train_dices.append(dice)


    NetG.train()
    train_IoUs = np.array(train_IoUs, dtype=np.float64)
    train_dices = np.array(train_dices, dtype=np.float64)
    train_mIoU = np.mean(train_IoUs, axis=0)
    train_mdice = np.mean(train_dices, axis=0)

    #val
    IoUs = np.array(IoUs, dtype=np.float64)
    dices = np.array(dices, dtype=np.float64)
    sens = np.array(sens, dtype=np.float64)
    pres = np.array(pres, dtype=np.float64)
    F1s = np.array(F1s, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mdice = np.mean(dices, axis=0)
    msen = np.mean(sens, axis=0)
    mpre = np.mean(pres, axis=0)
    mF1 = np.mean(F1s, axis=0)

    print('train_mIoU: {:.4f}'.format(train_mIoU))
    print('train_Dice: {:.4f}'.format(train_mdice))
    print('mIoU: {:.4f}'.format(mIoU))
    print('Dice: {:.4f}'.format(mdice))
    log['epoch'].append(epoch)
    log['train_iou'].append(train_mIoU)
    log['train_dice'].append(train_mdice)
    log['val_iou'].append(mIoU)
    log['val_dice'].append(mdice)
    log['G_Loss'].append(loss_G.data)
    
    for key in log:
        log[key] = [x.cpu().numpy() if torch.is_tensor(x) else x for x in log[key]]
    pd.DataFrame(log).to_csv('outputGANliver3/log.csv'
                                , index=False)

    if mIoU > max_iou:
        max_iou = mIoU
        best_dice = mdice
        best_sen = msen
        best_pre = mpre
        best_F1 = mF1
        torch.save(NetG.state_dict(), '%s/NetG_epoch_best.pth' % (config['outpath']))


print('IoU: %.4f' % max_iou)
print('Dice: %.4f' % best_dice)
print('sensitivity: %.4f' % best_sen)
print('precision: %.4f' % best_pre)
print('F1: %.4f' % best_F1)
