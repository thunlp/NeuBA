import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    """
    Model used to pretrain.
    Based on torchvision.models.densenet161
    """

    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.resnet = models.resnet152(pretrained=pretrained)
        # actually, this acts like the fc layer is removed.
        self.resnet.fc = nn.Sequential()
        self.resnet.layer4[2].relu = nn.Sequential()
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        embed = self.resnet(x)
        output = self.fc(embed)
        return output, embed


class DenseNet(nn.Module):
    """
    Model used to pretrain.
    Based on torchvision.models.densenet161
    """

    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.net = models.densenet201(pretrained=pretrained)
        # actually, this acts like the fc layer is removed.
        self.net.classifier = nn.Sequential()
        self.fc = nn.Linear(1920, num_classes, bias=True)

    def forward(self, x):
        embed = self.net(x)
        output = self.fc(embed)
        return output, embed
