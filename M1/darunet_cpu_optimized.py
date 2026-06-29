"""
CPU-Optimized DARU-Net for Maximum Test Accuracy
Optimized for CPU training with enhanced features for best performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CPUOptimizedCSAR(nn.Module):
    """CPU-Optimized Channel-Spatial Attention Residual Block"""
    
    def __init__(self, channel, reduction_ratio=8):
        super(CPUOptimizedCSAR, self).__init__()
        
        # Reduced reduction ratio for CPU efficiency
        reduction_ratio = min(reduction_ratio, max(4, channel // 8))
        
        # Channel Attention - optimized for CPU
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Simplified but effective channel attention
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False)
        )
        
        # Spatial attention - optimized for CPU
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # Efficient normalization
        self.norm = nn.BatchNorm2d(channel)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        
    def forward(self, x):
        residual = x
        
        # Pre-normalization
        x = self.norm(x)
        
        # Channel attention
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        # Apply dropout
        x = self.dropout(x)
        
        # Residual connection
        x = x + self.residual_conv(residual)
        
        return x

class CPUOptimizedConvBlock(nn.Module):
    """CPU-Optimized Convolution Block"""
    
    def __init__(self, ch_in, ch_out):
        super(CPUOptimizedConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class CPUOptimizedUpConv(nn.Module):
    """CPU-Optimized Up Convolution"""
    
    def __init__(self, ch_in, ch_out):
        super(CPUOptimizedUpConv, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.up(x)

class CPUOptimizedDARUNet(nn.Module):
    """CPU-Optimized DARU-Net for Maximum Test Accuracy"""
    
    def __init__(self, use_all_s2_channels=True):
        super(CPUOptimizedDARUNet, self).__init__()
        
        # Optimized filter configuration for CPU
        # Balanced between complexity and efficiency
        filters = [32, 64, 128, 256, 512]
        
        # Input channels
        s1_channels = 1  # Sentinel-1 VH polarization
        s2_channels = 12 if use_all_s2_channels else 4  # All bands or RGB+NIR
        
        # Encoder path for Sentinel-1
        self.s1_conv1 = CPUOptimizedConvBlock(s1_channels, filters[0])
        self.s1_csar1 = CPUOptimizedCSAR(filters[0])
        
        self.s1_conv2 = CPUOptimizedConvBlock(filters[0], filters[1])
        self.s1_csar2 = CPUOptimizedCSAR(filters[1])
        
        self.s1_conv3 = CPUOptimizedConvBlock(filters[1], filters[2])
        self.s1_csar3 = CPUOptimizedCSAR(filters[2])
        
        self.s1_conv4 = CPUOptimizedConvBlock(filters[2], filters[3])
        self.s1_csar4 = CPUOptimizedCSAR(filters[3])
        
        self.s1_conv5 = CPUOptimizedConvBlock(filters[3], filters[4])
        self.s1_csar5 = CPUOptimizedCSAR(filters[4])
        
        # Encoder path for Sentinel-2
        self.s2_conv1 = CPUOptimizedConvBlock(s2_channels, filters[0])
        self.s2_csar1 = CPUOptimizedCSAR(filters[0])
        
        self.s2_conv2 = CPUOptimizedConvBlock(filters[0], filters[1])
        self.s2_csar2 = CPUOptimizedCSAR(filters[1])
        
        self.s2_conv3 = CPUOptimizedConvBlock(filters[1], filters[2])
        self.s2_csar3 = CPUOptimizedCSAR(filters[2])
        
        self.s2_conv4 = CPUOptimizedConvBlock(filters[2], filters[3])
        self.s2_csar4 = CPUOptimizedCSAR(filters[3])
        
        self.s2_conv5 = CPUOptimizedConvBlock(filters[3], filters[4])
        self.s2_csar5 = CPUOptimizedCSAR(filters[4])
        
        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge - combining both paths
        bridge_channels = filters[4] * 2
        self.bridge_conv = CPUOptimizedConvBlock(bridge_channels, bridge_channels)
        self.bridge_csar = CPUOptimizedCSAR(bridge_channels)
        
        # Decoder path - simplified for CPU efficiency
        self.up5 = CPUOptimizedUpConv(bridge_channels, filters[4])
        self.up_conv5 = CPUOptimizedConvBlock(filters[4] + filters[3] * 2, filters[4])  # up + both skips
        self.up_csar5 = CPUOptimizedCSAR(filters[4])

        self.up4 = CPUOptimizedUpConv(filters[4], filters[3])
        self.up_conv4 = CPUOptimizedConvBlock(filters[3] + filters[2] * 2, filters[3])  # up + both skips
        self.up_csar4 = CPUOptimizedCSAR(filters[3])

        self.up3 = CPUOptimizedUpConv(filters[3], filters[2])
        self.up_conv3 = CPUOptimizedConvBlock(filters[2] + filters[1] * 2, filters[2])  # up + both skips
        self.up_csar3 = CPUOptimizedCSAR(filters[2])

        self.up2 = CPUOptimizedUpConv(filters[2], filters[1])
        self.up_conv2 = CPUOptimizedConvBlock(filters[1] + filters[0] * 2, filters[1])  # up + both skips
        self.up_csar2 = CPUOptimizedCSAR(filters[1])

        self.up1 = CPUOptimizedUpConv(filters[1], filters[0])
        self.up_conv1 = CPUOptimizedConvBlock(filters[0], filters[0])  # No skip connection at final level
        self.up_csar1 = CPUOptimizedCSAR(filters[0])
        
        # Final output layers
        self.final_conv = nn.Conv2d(filters[0], filters[0] // 2, kernel_size=3, padding=1, bias=False)
        self.final_norm = nn.BatchNorm2d(filters[0] // 2)
        self.final_act = nn.ReLU(inplace=True)
        self.final = nn.Conv2d(filters[0] // 2, 2, kernel_size=1)
        
        # Output activation
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # Store configuration
        self.filters = filters
        
    def forward(self, s1, s2):
        # Sentinel-1 encoder path
        s1_1 = self.s1_conv1(s1)
        s1_1 = self.s1_csar1(s1_1)
        s1_p1 = self.maxpool(s1_1)
        
        s1_2 = self.s1_conv2(s1_p1)
        s1_2 = self.s1_csar2(s1_2)
        s1_p2 = self.maxpool(s1_2)
        
        s1_3 = self.s1_conv3(s1_p2)
        s1_3 = self.s1_csar3(s1_3)
        s1_p3 = self.maxpool(s1_3)
        
        s1_4 = self.s1_conv4(s1_p3)
        s1_4 = self.s1_csar4(s1_4)
        s1_p4 = self.maxpool(s1_4)
        
        s1_5 = self.s1_conv5(s1_p4)
        s1_5 = self.s1_csar5(s1_5)
        
        # Sentinel-2 encoder path
        s2_1 = self.s2_conv1(s2)
        s2_1 = self.s2_csar1(s2_1)
        s2_p1 = self.maxpool(s2_1)
        
        s2_2 = self.s2_conv2(s2_p1)
        s2_2 = self.s2_csar2(s2_2)
        s2_p2 = self.maxpool(s2_2)
        
        s2_3 = self.s2_conv3(s2_p2)
        s2_3 = self.s2_csar3(s2_3)
        s2_p3 = self.maxpool(s2_3)
        
        s2_4 = self.s2_conv4(s2_p3)
        s2_4 = self.s2_csar4(s2_4)
        s2_p4 = self.maxpool(s2_4)
        
        s2_5 = self.s2_conv5(s2_p4)
        s2_5 = self.s2_csar5(s2_5)
        
        # Bridge - combine both paths
        bridge_input = torch.cat([s1_5, s2_5], dim=1)
        bridge = self.bridge_conv(bridge_input)
        bridge = self.bridge_csar(bridge)
        
        # Decoder path with skip connections
        d5 = self.up5(bridge)
        # Concatenate skip connections from both paths
        skip5 = torch.cat([s1_4, s2_4], dim=1)  # Combine both encoder paths
        d5 = torch.cat([d5, skip5], dim=1)
        d5 = self.up_conv5(d5)
        d5 = self.up_csar5(d5)

        d4 = self.up4(d5)
        skip4 = torch.cat([s1_3, s2_3], dim=1)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.up_conv4(d4)
        d4 = self.up_csar4(d4)

        d3 = self.up3(d4)
        skip3 = torch.cat([s1_2, s2_2], dim=1)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.up_conv3(d3)
        d3 = self.up_csar3(d3)

        d2 = self.up2(d3)
        skip2 = torch.cat([s1_1, s2_1], dim=1)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.up_conv2(d2)
        d2 = self.up_csar2(d2)

        d1 = self.up1(d2)
        d1 = self.up_conv1(d1)
        d1 = self.up_csar1(d1)
        
        # Final output
        out = self.final_conv(d1)
        out = self.final_norm(out)
        out = self.final_act(out)
        out = self.final(out)

        # Ensure output is 256x256 to match input
        if out.shape[-2:] != (256, 256):
            out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=True)

        # Apply log softmax
        output = self.log_softmax(out)

        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss for better training"""
    
    def __init__(self, focal_weight=0.7, dice_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def dice_loss(self, inputs, targets):
        """Dice loss for segmentation"""
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :]  # Get positive class
        targets = targets.float()
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        return 1 - dice
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice
