import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, blur=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch//2 + skip_ch, out_ch)
        
    def forward(self, x, skip_x):
        x = self.up(x)
        # Pad if needed, but normally aligned
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    """
    U-Net architecture with ResNet34 encoder for Transfer Learning.
    Matches the architecture style used in DustScratchRemoval.
    """
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        # Load pretrained ResNet34
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Encoder (Freezing initially can be handled in training script)
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # out: 64, H/4, W/4
        self.layer1 = resnet.layer1 # out: 64, H/4, W/4
        self.layer2 = resnet.layer2 # out: 128, H/8, W/8
        self.layer3 = resnet.layer3 # out: 256, H/16, W/16
        self.layer4 = resnet.layer4 # out: 512, H/32, W/32
        
        # Decoder
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up3 = UpBlock(128, 64, 64)
        
        # Since inc downsamples by 4x initially, we need an extra upsampling block to get back to H, W
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(64, 32)
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1 = self.layer1(x1) # Skip 1: 64 ch
        x2 = self.layer2(x1) # Skip 2: 128 ch
        x3 = self.layer3(x2) # Skip 3: 256 ch
        x4 = self.layer4(x3) # Bottleneck: 512 ch
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Extra upsampling
        x = self.up4(x)
        x = self.up_conv4(x)
        x = self.up5(x)
        
        x = self.outc(x)
        return self.sigmoid(x)
