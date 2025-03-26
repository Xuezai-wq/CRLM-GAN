import torch
import torch.nn as nn
from torchvision import models

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        # features = models.resnet50(pretrained=True).features
        model = models.resnet50(pretrained=True)
        resnet50_feature_extractor = nn.Sequential(*list(model.children()))
        self.to_relu_1_2 = nn.Sequential()
        # self.to_relu_2_2 = nn.Sequential()
        # self.to_relu_3_3 = nn.Sequential()
        # self.to_relu_4_3 = nn.Sequential()
        for x in range(5):
            self.to_relu_1_2.add_module(str(x), resnet50_feature_extractor[x:x+1])
        # for x in range(5, 6):
            self.to_relu_2_2=resnet50_feature_extractor[5:6]
        # for x in range(6, 7):
            self.to_relu_3_3=resnet50_feature_extractor[6:7]
        # for x in range(7, 8):
            self.to_relu_4_3=resnet50_feature_extractor[7:8]
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


# a=torch.rand(8,3,224,224)
# model= Resnet50()
#
# print(torch.cat((torch.tensor(model(a)[0]).view(8,-1),torch.tensor(model(a)[1]).view(8,-1),torch.tensor(model(a)[2]).view(8,-1),torch.tensor(model(a)[3]).view(8,-1)),1).shape)
