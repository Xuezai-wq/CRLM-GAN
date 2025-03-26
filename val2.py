import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics2 import *
from utils import AverageMeter
import numpy as np
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'testing/images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    #_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids = img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    '''val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])'''
    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'testing/images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'testing/masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter(),
                  'F1': AverageMeter()}
    iou_values = []
    dice_values = []
    sensitivity_values = []
    precision_values = []
    F1_values = []

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            sensitivity = get_sensitivity(output, target)
            specificity = get_specificity(output, target)
            precision = get_precision(output, target)
            F1 = get_F1(output, target)
            iou_values.append(iou)
            dice_values.append(dice)
            sensitivity_values.append(sensitivity)
            precision_values.append(precision)
            F1_values.append(F1)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['sensitivity'].update(sensitivity, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))


            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meters['iou'].avg)
    print('Dice: %.4f' % avg_meters['dice'].avg)
    print('sensitivity: %.4f' % avg_meters['sensitivity'].avg)
    print('specificity: %.4f' % avg_meters['specificity'].avg)
    print('precision: %.4f' % avg_meters['precision'].avg)
    print('F1: %.4f' % avg_meters['F1'].avg)
    
    dice_values = np.array(dice_values)

    # 计算Dice系数的均值和标准误差
    mean_dice = np.mean(dice_values)
    std_dice = np.std(dice_values)
    n = len(dice_values)

    # 计算95%置信区间
    z = stats.norm.ppf(0.975)  # 95%置信区间的Z值
    se = std_dice / np.sqrt(n)  # 标准误差

    ci_lower = mean_dice - z * se  # 置信区间下界
    ci_upper = mean_dice + z * se  # 置信区间上界

    # 打印置信区间
    print(f"Dice系数的95%置信区间: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    compute_confidence_intervals(iou_values, dice_values, sensitivity_values, precision_values, F1_values)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
