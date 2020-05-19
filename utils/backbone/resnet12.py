import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


ceil = True
inp =True
out_dim = 512

class Selayer(nn.Module):

    def __init__(self, inplanes):
        super(Selayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(inplanes // 16, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)

        self.convr = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, padding=1)
        self.bnr = nn.BatchNorm2d(planes, eps=2e-5)

        self.relu = nn.ReLU(inplace=inp)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil)


    def forward(self, x):
        identity = self.convr(x)
        identity = self.bnr(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(type(out))
        # print(out.size())
        # out = self.selayer(out)

        out += identity
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class ResNet12(nn.Module):
    def __init__(self, drop_ratio=0.1, with_drop=False):
        super(ResNet12, self).__init__()

        self.drop_layers = with_drop
        self.inplanes = 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.layer1 = self._make_layer(ResNetBlock, 64)
        self.layer2 = self._make_layer(ResNetBlock, 128)
        self.layer3 = self._make_layer(ResNetBlock, 256)
        self.layer4 = self._make_layer(ResNetBlock, 512)

        # global weight
        self.weight = nn.Linear(out_dim, 64)
        nn.init.xavier_uniform_(self.weight.weight)
        self.dropout = nn.Dropout(drop_ratio, inplace=inp)
        # length scale parameters
        # self.conv1_ls = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        # self.bn1_ls = nn.BatchNorm2d(512, eps=2e-5)
        self.conv1_ls = nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3)
        self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        self.fc1_ls = nn.Linear(16, 1)

        # self.classifier_test = nn.Linear(512, 5)
        # self.classifier = nn.Linear(512, 64)
        # self.classifier.bias.data.fill_(0)
        self.relu = nn.ReLU(inplace=inp)
        self.pool = nn.AvgPool2d(7)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='conv2d')
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        x = self.layer1(x)
        x = self.dropout(x)
        # print(x.size())
        x = self.layer2(x)
        x = self.dropout(x)
        # print(x.size())
        x = self.layer3(x)
        x3 = self.dropout(x)
        # print(x3.size())
        x = self.layer4(x3)
        x4 = self.dropout(x)
        # print(x4.size())
        if self.drop_layers:
            return [x4, x3]
        else:
            return [x4]