import torch
import torch.nn as nn

inp = False

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.weight = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.weight.weight)

        # length scale parameters
        # self.conv1_ls = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        # self.bn1_ls = nn.BatchNorm2d(512, eps=2e-5)
        self.conv1_ls = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3)
        self.bn1_ls = nn.BatchNorm2d(1, eps=2e-5)
        self.fc1_ls = nn.Linear(16, 1)

        # self.classifier_test = nn.Linear(512, 5)
        # self.classifier = nn.Linear(512, 64)
        # self.classifier.bias.data.fill_(0)
        self.relu = nn.ReLU(inplace=inp)
        self.pool = nn.AvgPool2d(7)

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),  # inplace为False，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),

        )
        self.feature2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),)

        self.dropout = nn.Dropout(0.1, inplace=False)

    def forward(self, x):
        x = self.feature(x)
        x = self.dropout(x)
        x2 = self.feature2(x)
        x2 = self.dropout(x2)
        # print(x2.size())
        return [x2]


