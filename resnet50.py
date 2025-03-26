from torchvision import models
import torch
from torch import nn
# 加载精度为76.130%的旧权重参数文件V1
# model_v1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# 等价写法
# model_v1 = models.resnet50(weights="IMAGENET1K_V1")
#
#
# print(model_v1)
a=torch.rand(8,3,20,20)
model = models.resnet50(pretrained=True)
resnet50_feature_extractor=nn.Sequential(*list(model.children()))
for i in resnet50_feature_extractor:
    print(i)
    print('-' * 50)


# features= model.features
#
# print(features)
# print(model)
# print(model.layer4)
# # 加载精度为80.858%的新权重参数文件V2
# model_v2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# # 等价写法
# model_v1 = models.resnet50(weights="IMAGENET1K_V2")
# output=model(a)
# print(output.layer3)