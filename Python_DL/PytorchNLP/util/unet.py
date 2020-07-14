import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation

        :param in_channels: (int) Number of input channels
        :param n_classes: (int) Number of output channels
        :param depth: (int) Depth of the network
        :param wf: (int) number of filters in the first layer is 2 ** wf
        :param padding: (bool) if True, apply padding such that the input shape
                        is the same as the output
        :param batch_norm: (bool) use BatchNorm after layers with an activation function
        :param up_mode: (str) one of 'upconv' or 'upsample'.
                        'upconv' will use transposed covolutions for learned upsampling
                        'upsample' will use bilinear upsampling
        """

        super (UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i), padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()

        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode, padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []

        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        blocks = []

        self.b_norm = batch_norm

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding))
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding))
        self.act2 = nn.ReLU()

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(x)

        if self.b_norm:
            out = self.batch_norm(out)

        out = self.conv2(out)
        out = self.act2(out)

        if self.b_norm:
            out = self.batch_norm(out)

        return out

class UNetUpBlock(nn.Module):
    def __int__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, height, width = layer.size()
        diff_x = (width - target_size[1]) // 2
        diff_y = (height - target_size[0]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

