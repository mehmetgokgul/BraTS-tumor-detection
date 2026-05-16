import torch.nn as nn
from Convolution import Convolution
from Decoder import Decoder


class MyUNet(nn.Module):

    def __init__(self, in_channels=4, num_classes=4):
        super(MyUNet, self).__init__()

        # --- ENCODER ---
        self.c1 = Convolution(in_channels, 64)    # çıkış: 64
        self.c2 = Convolution(64, 128)             # çıkış: 128
        self.c3 = Convolution(128, 256)            # çıkış: 256

        # --- BOTTLENECK (dropout burada anlamlı) ---
        self.bn = Convolution(256, 512, dropout=0.3)  # çıkış: 512

        # --- DECODER ---
        # d1: bn(512) + skip c3(256) = 768 giriş → 256 çıkış
        self.d1 = Decoder(512, 256, skip_channels=256)
        # d2: d1(256) + skip c2(128) = 384 giriş → 128 çıkış
        self.d2 = Decoder(256, 128, skip_channels=128)
        # d3: d2(128) + skip c1(64)  = 192 giriş → 64 çıkış
        self.d3 = Decoder(128, 64,  skip_channels=64)

        # --- FINAL ---
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        f1 = self.c1(x)
        p1 = self.pool(f1)

        f2 = self.c2(p1)
        p2 = self.pool(f2)

        f3 = self.c3(p2)
        p3 = self.pool(f3)

        bn_out = self.bn(p3)

        u1 = self.d1(bn_out, f3)
        u2 = self.d2(u1, f2)
        u3 = self.d3(u2, f1)

        return self.final(u3)