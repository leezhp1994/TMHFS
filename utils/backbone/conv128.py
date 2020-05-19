import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )


class ConvNet(nn.Module):

    def __init__(self, with_drop=False, x_dim=3, hid_dim=64, z_dim=128):
        super().__init__()

        self.drop_layer = with_drop

        self.layer1 = conv_block(x_dim,   hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block(hid_dim, z_dim)
        self.layer4 = conv_block(z_dim,   z_dim)

        self.weight = nn.Linear(z_dim, 64)
        nn.init.xavier_uniform_(self.weight.weight)

        self.conv1_pt = nn.Conv2d(in_channels=z_dim, out_channels=1, kernel_size=3)
        self.bn1_pt = nn.BatchNorm2d(1, eps=2e-5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1_pt = nn.Linear(16, 1)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.drop_layer:
            return [x4, x3]
        else:
            return [x4]