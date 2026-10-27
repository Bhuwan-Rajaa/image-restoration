"""Pixel and structural reconstruction, plus optional normalized VGG features."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16


def structural_similarity(prediction, target):
    """Per-image RGB SSIM, 11x11 Gaussian valid window, data range 1.

    Computes local statistics in float32 even under autocast. Validation images
    use this same definition consistently; legacy skimage scores may differ.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, target = prediction.float(), target.float()
        coordinates = torch.arange(11, device=prediction.device, dtype=torch.float32) - 5
        gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
        gaussian = gaussian / gaussian.sum()
        window = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11).contiguous()
        moments = F.conv2d(torch.cat((prediction, target, prediction.square(), target.square(),
                                     prediction * target)), window, groups=3)
        mean_x, mean_y, second_x, second_y, cross = moments.chunk(5)
        variance_x = (second_x - mean_x.square()).clamp_min(0)
        variance_y = (second_y - mean_y.square()).clamp_min(0)
        covariance = cross - mean_x * mean_y
        numerator = (2 * mean_x * mean_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
        denominator = (mean_x.square() + mean_y.square() + 0.01 ** 2) * (
            variance_x + variance_y + 0.03 ** 2)
        return (numerator / denominator).mean(dim=(1, 2, 3))


class VGGFeatureLoss(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # relu1_2, relu2_2, relu3_3, before pooling; include shallow fine detail.
        features = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None).features
        self.features = features[:16].eval().requires_grad_(False)
        self.layer_ids = (3, 8, 15)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, prediction, target):
        x = (prediction - self.mean) / self.std
        y = (target - self.mean) / self.std
        total = prediction.new_zeros((), dtype=torch.float32)
        for i, layer in enumerate(self.features):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            if i in self.layer_ids:
                total = total + F.l1_loss(x.float(), y.float()) / len(self.layer_ids)
        return total


class RestorationLoss(nn.Module):
    def __init__(self, ssim_weight=0.2, perceptual=False, pretrained=True):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.vgg = VGGFeatureLoss(pretrained=pretrained) if perceptual else None

    def forward(self, prediction, target, perceptual_weight=0.0):
        pixel = F.l1_loss(prediction.float(), target.float())
        structural = 1 - structural_similarity(prediction, target).mean()
        feature = self.vgg(prediction, target) if self.vgg is not None and perceptual_weight else pixel.new_zeros(())
        loss = pixel + self.ssim_weight * structural + perceptual_weight * feature
        return loss, {'l1': pixel.detach(), 'ssim_loss': structural.detach(),
                      'vgg': feature.detach(), 'vgg_weighted': feature.detach() * perceptual_weight}


class MaskedRestorationLoss(RestorationLoss):
    """Add per-image damage-region supervision, including very small hair masks.

    Masks are training targets only; inference still takes a damaged RGB image.
    Empty masks (clean identity examples) contribute zero region loss.
    """
    def __init__(self, ssim_weight=0.2, perceptual=True, mask_weight=0.1, pretrained=True):
        super().__init__(ssim_weight, perceptual, pretrained)
        self.mask_weight = mask_weight

    def forward(self, prediction, target, mask, perceptual_weight=0.02):
        loss, components = super().forward(prediction, target, perceptual_weight)
        error = (prediction.float() - target.float()).abs()
        per_image = (error * mask).sum(dim=(1, 2, 3)) / (
            mask.sum(dim=(1, 2, 3)) * prediction.shape[1]).clamp_min(1)
        region = per_image.mean()
        components['mask_l1'] = region.detach()
        return loss + self.mask_weight * region, components
