import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channels, out_channels, kernel_size=3):
    stride=1
    padding=kernel_size//2
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
    )

class upResBlock(nn.Module):
  def __init__(self, in_channels, kernel_size=3):
    super().__init__()
    self.cbr=conv_bn_relu(in_channels,2*in_channels,kernel_size)
  def forward(self, x):
    return self.cbr(x)

class dwResBlock(nn.Module):
  def __init__(self, in_channels, kernel_size=3):
    super().__init__()
    self.cbr=conv_bn_relu(in_channels,in_channels//2,kernel_size)
  def forward(self, x):
    return self.cbr(x)

def down_pooling():
    return nn.MaxPool3d(2)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet3d(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = 1
        nclasses = 1
        # go down
        self.conv0 = conv_bn_relu(input_channels, 64)
        self.conv1 = upResBlock(64) 
        self.conv2 = upResBlock(128) 

        self.down_pooling = nn.MaxPool3d(2)

        # go up
        self.up_pool6 = up_pooling(256, 128)
        self.conv7 = dwResBlock(256) 
        self.up_pool8 = up_pooling(128, 64)
        self.conv9 = dwResBlock(128) 

        self.conv10 = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        # normalize input data
        # go down
        x0 = self.conv0(x)
        p0 = self.down_pooling(x0)
        x1 = self.conv1(p0)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)

        x6 = x2

        # go up
        p7 = self.up_pool6(x6)
        x7 = torch.cat([p7, x1], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x0], dim=1)
        x8 = self.conv9(x8)

        output = F.pad(self.conv10(x8), [0,0, 0,0, 0,0, 1,0])
        return output
