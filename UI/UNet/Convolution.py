import torch.nn as nn


class Convolution(nn.Module):

    def __init__(self, in_channels, filter_num, dropout=0.0):
        super(Convolution, self).__init__()

        layers = [
            nn.Conv2d(in_channels, filter_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_num, filter_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
        ]


        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)