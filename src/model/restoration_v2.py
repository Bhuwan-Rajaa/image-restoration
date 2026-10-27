"""ResNet34 restoration U-Net with full-resolution skips and residual output.

Derived from this project's ResNet U-Net / DustScratchRemoval approach.
New checkpoints are versioned; legacy weights must not be loaded as v2 weights.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


class DetailBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        # No batch statistics in the decoder: batches can be small on 6 GB GPUs.
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class ResizeBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = DetailBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(self.reduce(x), size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class RestorationUNet(nn.Module):
    architecture = 'restoration_v2'

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.ModuleDict({
            'stem': nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            'pool': backbone.maxpool,
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4,
        })
        self.detail = DetailBlock(3, 16)
        self.up1 = ResizeBlock(512, 256, 128)
        self.up2 = ResizeBlock(128, 128, 64)
        self.up3 = ResizeBlock(64, 64, 64)
        self.up4 = ResizeBlock(64, 64, 32)
        self.up5 = ResizeBlock(32, 16, 16)
        self.correction = nn.Conv2d(16, 3, 3, padding=1)
        # Begin as the identity mapping, retaining every undamaged input pixel.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder_frozen = False

    def freeze_encoder(self, frozen):
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep ImageNet running statistics fixed, including when fine-tuning.
        # Affine parameters still learn after unfreezing. Avoid small-batch drift.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, image):
        detail = self.detail(image)
        half = self.encoder['stem']((image - self.mean) / self.std)
        quarter = self.encoder['layer1'](self.encoder['pool'](half))
        eighth = self.encoder['layer2'](quarter)
        sixteenth = self.encoder['layer3'](eighth)
        deep = self.encoder['layer4'](sixteenth)
        x = self.up1(deep, sixteenth)
        x = self.up2(x, eighth)
        x = self.up3(x, quarter)
        x = self.up4(x, half)
        x = self.up5(x, detail)
        # Do not clamp while optimizing: clipping can suppress corrective gradients.
        return image + self.correction(x)


def load_restoration_model(checkpoint, device='cpu'):
    """Load v2 training snapshots or historical raw state dictionaries."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') == RestorationUNet.architecture:
        model = RestorationUNet(pretrained=False)
        model.load_state_dict(saved['model'])
    elif 'architecture' in saved or 'model' in saved:
        raise ValueError('Unsupported checkpoint architecture')
    else:
        from unet import ResNetUNet
        model = ResNetUNet(pretrained=False)
        model.load_state_dict(saved)
    return model.to(device).eval()
