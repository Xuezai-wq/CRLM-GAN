import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def get_sen(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    output_set=output_.sum()
    target_set=target_.sum()

    return (intersection + smooth) / (target_set+ smooth)

def get_sensitivity(output, gt): # 求敏感度 se=TP/(TP+FN)
    SE = 0.
    if torch.is_tensor(output):
        output = torch.sigmoid(output)
    
    device = output.device
    gt = gt.to(device)

    output = output > 0.5
    gt = gt > 0.5

    TP = ((output == 1).byte() + (gt == 1).byte()) == 2
    FN = ((output==0).byte() + (gt==1).byte()) == 2
    #wfy:batch_num>1时，改进
    # if len(output)>1:
    #     for i in range(len(output)):
    #         SE += float(torch.sum(TP[i])) / (float(torch.sum(TP[i]+FN[i])) + 1e-5)
    # else:
    #     SE = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-5) #原本只用这一句
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-5)  # 原本只用这一句

    return SE #返回batch中所有样本的SE和

def get_specificity(SR, GT, threshold=0.5):#求特异性 sp=TN/(FP+TN)
    if torch.is_tensor(SR):
        SR = torch.sigmoid(SR)
    SR = SR > threshold #得到true和false
    GT = GT > threshold
    SP=0.# wfy
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    #wfy:batch_num>1时，改进
    # if len(SR)>1:
    #     for i in range(len(SR)):
    #         SP += float(torch.sum(TN[i])) / (float(torch.sum(TN[i] + FP[i])) + 1e-5)
    # else:
    #     SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-5) # 原本只用这一句
    #
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-5)
    return SP


def get_precision(output, target): #阳性预测值，准确率（precision）pr = TP/(TP+FP)
    smooth = 1e-5
    
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    '''if torch.is_tensor(output):
        output = torch.sigmoid(output)  # 应用 sigmoid
        device = output.device  # 获取设备
        target = target.to(device)  # 将 target 转移到相同设备

        # 转为 NumPy 数组以进行逻辑操作
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()'''
    
    
    output_ = output > 0.5
    target_ = target > 0.5
    ppv=0.
    # if len(output)>1:
    #     for i in range(len(output)):
    #         intersection = (output[i] * target[i]).sum()
    #         ppv += (intersection + smooth)/(output[i].sum() + smooth)
    # else:
    intersection = (output_ & target_).sum() # 一个数字,=TP
    ppv = (intersection + smooth)/(output_.sum() + smooth)

    # intersection = (output * target).sum() # TP
    return ppv

def get_F1(output, gt):
    se = get_sensitivity(output, gt)
    pc = get_precision(output, gt)
    f1 = 2*se*pc / (se+pc+1e-5)
    return f1




# a=np.array([1,1,0,0])
# b=np.array([0.6,.6,0,0])
# sen(a,b)

def compute_confidence_intervals(iou_values, dice_values, sensitivity_values, precision_values, F1_values):
    # 定义一个计算均值、标准误差和置信区间的辅助函数
    def calculate_confidence_interval(values):
        values = np.array(values)
        mean_value = np.mean(values)
        std_value = np.std(values)
        n = len(values)
        se = std_value / np.sqrt(n)
        z = stats.norm.ppf(0.975)  # 95%置信区间的Z值
        ci_lower = mean_value - z * se
        ci_upper = mean_value + z * se
        return mean_value, std_value, ci_lower, ci_upper

    # 对每个评估指标计算均值、标准误差和置信区间
    metrics = {
        "IoU": calculate_confidence_interval(iou_values),
        "Dice": calculate_confidence_interval(dice_values),
        "Sensitivity": calculate_confidence_interval(sensitivity_values),
        #"Specificity": calculate_confidence_interval(specificity_values),
        "Precision": calculate_confidence_interval(precision_values),
        "F1": calculate_confidence_interval(F1_values)
    }

    # 打印结果
    for metric, (mean_value, std_value, ci_lower, ci_upper) in metrics.items():
        #print(f"{metric}的均值: {mean_value:.4f}, 标准差: {std_value:.4f}")
        print(f"{metric}的95%置信区间: ({ci_lower:.4f}, {ci_upper:.4f})\n")
