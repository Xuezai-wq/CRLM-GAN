import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'TverskyLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        # pred = input.squeeze(dim=1)

        # smooth = 1
        #
        # # dice系数的定义
        # dice1 = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) +
        #                                                            0.3 * (pred * (1 - target)).sum(dim=1).sum(
        #             dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        #
        # Tversky=torch.clamp((1 - dice1).mean(), 0, 2)

        # return 0.5 * bce + dice

        return 0.5 * bce + dice


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#
#         return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


# class TverskyLoss(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target):
#         pred = pred.squeeze(dim=1)
#
#         smooth = 1
#
#         # dice系数的定义
#         dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) +
#                                                                    0.3 * (pred * (1 - target)).sum(dim=1).sum(
#                     dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
#
#         # 返回的是dice距离
#         return torch.clamp((1 - dice).mean(), 0, 2)


def tversky_loss(true, inputs, alpha, beta, eps=1e-5):
    y_true_pos = inputs.contiguous().view(-1)
    y_pred_pos = true.contiguous().view(-1)
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum()
    false_pos = ((1-y_true_pos) * y_pred_pos).sum()
    score = (true_pos + eps) / (true_pos + alpha*false_neg + beta*false_pos + eps)
    return (1 - score)

class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.name = f'Tversky(a={beta})'
        # self.alpha = 1 - beta
        # self.beta = beta

    def forward(self, inputs, true):
        return tversky_loss(true, inputs, 0.3, 0.7)