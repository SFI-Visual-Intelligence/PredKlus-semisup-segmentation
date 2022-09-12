import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool (maxpool at U-Net class).
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.main = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            conv3x3(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        before_pool = self.main(x)
        return before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.main = nn.Sequential(
            conv3x3(2 * self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            conv3x3(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        x = torch.cat((from_up, from_down), 1)
        x = self.main(x)
        return x

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the transpose convolution (specified by upmode='transpose')
    """
    valid = False
    pad  = 0
    fow = [192, 192]
    dim = 2
    type = 'seg'
    stride = 1
    increase_fow = 16

    def __init__(self, n_classes=2, in_channels=1, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)

            down_conv = DownConv(ins, outs)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.Sequential(*self.down_convs)
        self.up_convs = nn.Sequential(*self.up_convs)
        self.conv_final = nn.Sequential(
            conv1x1(self.start_filts, n_classes),
            nn.LogSoftmax(dim=1)
        )

        # Todo: Apply initialization?
        #self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            #nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            print(i, m)
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x = module(x)  # before pool
            encoder_outs.append(x)
            if i < self.depth - 1:               # pooling
                x = self.pool(x)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        if self.conv_final is None:
            return x
        else:
            x = self.conv_final(x)
            return x


if __name__ == "__main__":
    model = UNet(n_classes=3, in_channels=4)
    print(model)
