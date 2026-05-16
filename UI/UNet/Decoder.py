import torch
import torch.nn as nn
import torch.nn.functional as F
from Convolution import Convolution


class Decoder(nn.Module):

    def __init__(self, in_channels, filter_num, skip_channels):
        super(Decoder, self).__init__()
        self.up         = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # concat sonrası gerçek kanal sayısı: in_channels + skip_channels
        self.conv_block = Convolution(in_channels + skip_channels, filter_num)

    def forward(self, current_input, skip_connection):
        up_img = self.up(current_input)
        # Boyut farkını kapat
        if up_img.shape[2:] != skip_connection.shape[2:]:
            up_img = F.interpolate(up_img, size=skip_connection.shape[2:],
                                   mode="bilinear", align_corners=True)
        merged = torch.cat([up_img, skip_connection], dim=1)
        return self.conv_block(merged)