import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

class FeatureLoss(nn.Module):
    """
    Perceptual Loss matching the DustScratchRemoval transfer learning approach.
    Uses pretrained VGG16 layers to compute feature differences.
    """
    def __init__(self, layer_wgts=[0, 0, 20, 70, 10]):
        super().__init__()
        # VGG16 features
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        # The layer ids from DustScratchRemoval (blocks ending in MaxPool2d)
        # Typically these map to indices: 4, 9, 16, 23, 30
        self.layer_ids = [4, 9, 16, 23, 30] 
        self.wgts = layer_wgts
        
        self.slices = nn.ModuleList()
        start = 0
        for idx in self.layer_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[start:idx+1]))
            start = idx + 1
            
        self.base_loss = nn.L1Loss()
        
    def _extract_features(self, x):
        features = []
        for slice_module in self.slices:
            x = slice_module(x)
            features.append(x)
        return features

    def _gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1,2)) / (c * h * w)

    def forward(self, input, target):
        in_feat = self._extract_features(input)
        out_feat = self._extract_features(target)
        
        loss = self.base_loss(input, target) # Base L1 Loss
        
        for f_in, f_out, w in zip(in_feat, out_feat, self.wgts):
            if w > 0:
                loss += self.base_loss(f_in, f_out) * w
                # Style loss component
                loss += self.base_loss(self._gram_matrix(f_in), self._gram_matrix(f_out)) * (w ** 2) * 5e3
                
        return loss

def gaussian(window_size, sigma):
    import math
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    import math
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    # Simplified SSIM implementation
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        return 1.0 - ssim(input, target)

class CombinedLoss(nn.Module):
    def __init__(self, mode='baseline'):
        super().__init__()
        self.mode = mode
        self.l1 = nn.L1Loss()
        
        if mode == 'perceptual1':
            # Model 1 weights from original repo
            self.feat_loss = FeatureLoss(layer_wgts=[0, 0, 20, 70, 10])
        elif mode == 'perceptual2':
            # Model 2 experimental weights
            self.feat_loss = FeatureLoss(layer_wgts=[20, 20, 20, 20, 20])
        elif mode == 'ssim':
            self.feat_loss = FeatureLoss(layer_wgts=[0, 0, 20, 70, 10])
            self.ssim_loss = SSIMLoss()

    def forward(self, input, target):
        if self.mode == 'baseline':
            return self.l1(input, target)
        elif self.mode in ['perceptual1', 'perceptual2']:
            return self.feat_loss(input, target)
        elif self.mode == 'ssim':
            # Perceptual + SSIM
            loss_feat = self.feat_loss(input, target)
            loss_ssim = self.ssim_loss(input, target)
            return loss_feat + 0.1 * loss_ssim
