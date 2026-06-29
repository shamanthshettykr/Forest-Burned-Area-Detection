import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()

        # Higher dropout rate for better regularization
        dropout_rate = 0.2

        # First convolution with Group Normalization (more stable than BatchNorm)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # LeakyReLU for better gradient flow

        # Use Dropout2d for spatial coherence in feature maps
        self.drop1 = nn.Dropout2d(dropout_rate)

        # Second convolution with residual connection
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop2 = nn.Dropout2d(dropout_rate)

        # Enhanced residual connection with normalization
        self.residual = nn.Identity()
        if ch_in != ch_out:
            self.residual = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out)
            )

    def forward(self, x):
        residual = self.residual(x)

        # First convolution block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual  # Residual connection
        x = self.act2(x)
        x = self.drop2(x)

        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()

        # Higher dropout rate for better regularization
        dropout_rate = 0.2

        # Advanced upsampling with transposed convolution for better feature learning
        self.up_conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False)

        # Group Normalization with more groups for better feature normalization
        self.norm = nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out)

        # LeakyReLU for better gradient flow
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Dropout2d for spatial coherence
        self.drop = nn.Dropout2d(dropout_rate)

        # Add residual connection for better gradient flow
        self.has_residual = True
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(32, ch_out), num_channels=ch_out)
        )

    def forward(self, x):
        # Main path
        out = self.up_conv(x)
        out = self.norm(out)

        # Add residual connection
        if self.has_residual:
            residual = self.residual(x)
            out = out + residual

        out = self.act(out)
        out = self.drop(out)

        return out

class CSAR_Block(nn.Module):
    """
    Enhanced Channel-Spatial Attention Residual Block with increased complexity
    Combines both average and max pooling for better feature representation
    """
    def __init__(self, channel, reduction_ratio=16, enhanced=True):
        super(CSAR_Block, self).__init__()

        # Adaptive reduction ratio based on channel size
        reduction_ratio = min(reduction_ratio, max(8, channel // 16))
        self.enhanced = enhanced

        # Channel Attention - use both average and max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Enhanced channel attention with multiple layers for better feature learning
        if enhanced:
            self.channel_attention = nn.Sequential(
                nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False),
                nn.BatchNorm2d(channel // reduction_ratio),  # Use BatchNorm instead of GroupNorm
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel // reduction_ratio, channel // (reduction_ratio // 2), kernel_size=1, bias=False),
                nn.BatchNorm2d(channel // (reduction_ratio // 2)),  # Use BatchNorm instead of GroupNorm
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel // (reduction_ratio // 2), channel, kernel_size=1, bias=False)
            )
        else:
            self.channel_attention = nn.Sequential(
                nn.Conv2d(channel, channel // reduction_ratio, kernel_size=1, bias=False),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel // reduction_ratio, channel, kernel_size=1, bias=False)
            )

        # Enhanced spatial attention with multiple convolution layers
        if enhanced:
            self.conv_spatial = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(8),  # Use BatchNorm instead of GroupNorm
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(8, 4, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(4),  # Use BatchNorm instead of GroupNorm
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
            )
        else:
            self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # Adaptive dropout rate
        dropout_rate = 0.1 if enhanced else 0.2
        self.dropout = nn.Dropout2d(dropout_rate)

        # Enhanced residual connection with pre-normalization
        self.pre_norm = nn.BatchNorm2d(channel)  # Use BatchNorm instead of GroupNorm

        # Multi-scale residual connections for enhanced complexity
        if enhanced:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(channel),  # Use BatchNorm instead of GroupNorm
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
            )
        else:
            self.residual_conv = nn.Conv2d(channel, channel, kernel_size=1, bias=False)

        self.post_norm = nn.BatchNorm2d(channel)  # Use BatchNorm instead of GroupNorm

    def forward(self, x):
        residual = x

        # Channel Attention - using both avg and max pooling
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)

        x = x * channel_out

        # Spatial Attention - using both avg and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = torch.sigmoid(self.conv_spatial(spatial_out))

        x = x * spatial_out

        # Apply dropout
        x = self.dropout(x)

        # Enhanced Residual Connection with pre-normalization
        x = self.pre_norm(x)
        x = self.residual_conv(x)
        x = x + residual
        x = self.post_norm(x)

        return x

class PaperL2Loss(nn.Module):
    """
    L2 Loss function as described in the DARU-Net paper:
    L(θ) = 1/M * Σ[fθ(S1(i), S2(i)) − R(i)]²

    This loss function works with log_softmax outputs and binary targets.
    """
    def __init__(self):
        super(PaperL2Loss, self).__init__()

    def forward(self, log_probs, targets):
        """
        Args:
            log_probs: [B, 2, H, W] tensor of log probabilities from model
            targets: [B, H, W] or [B, 1, H, W] tensor of binary ground truth (0 or 1)
        """
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs)

        # Get the probability of the positive class (burned area)
        pred_burned = probs[:, 1, :, :]  # [B, H, W]

        # Convert targets to float and ensure same shape
        targets = targets.float()
        if targets.dim() == 4:  # [B, 1, H, W]
            targets = targets.squeeze(1)  # [B, H, W]

        # Ensure spatial dimensions match by resizing if necessary
        if pred_burned.shape[-2:] != targets.shape[-2:]:
            targets = nn.functional.interpolate(
                targets.unsqueeze(1),
                size=pred_burned.shape[-2:],
                mode='nearest'
            ).squeeze(1)

        # Calculate L2 loss as per paper equation (2)
        M = targets.numel()  # Total number of pixels
        loss = torch.sum((pred_burned - targets) ** 2) / M

        return loss

class DARU_Net(nn.Module):
    def __init__(self, use_paper_config=True, use_log_softmax=True, enhanced_complexity=True):
        super(DARU_Net, self).__init__()

        # Configuration based on paper specifications with enhanced complexity option
        if use_paper_config:
            # Paper configuration: 16, 32, 64, 128, 256 filters
            base_filters = [16, 32, 64, 128, 256]
            s2_channels = 12  # Use all 12 Sentinel-2 channels for maximum accuracy
        else:
            # Enhanced configuration: 32, 64, 128, 256, 512 filters
            base_filters = [32, 64, 128, 256, 512]
            s2_channels = 12  # All Sentinel-2 channels

        # Increase complexity for better accuracy
        if enhanced_complexity:
            # Double the filter sizes for increased model capacity
            filters = [f * 2 for f in base_filters]
        else:
            filters = base_filters

        self.use_log_softmax = use_log_softmax
        self.enhanced_complexity = enhanced_complexity

        # Encoder path for Sentinel-1 (5 levels as per paper)
        self.s1_conv1 = conv_block(ch_in=1, ch_out=filters[0])
        self.s1_csar1 = CSAR_Block(channel=filters[0], enhanced=enhanced_complexity)

        self.s1_conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.s1_csar2 = CSAR_Block(channel=filters[1], enhanced=enhanced_complexity)

        self.s1_conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.s1_csar3 = CSAR_Block(channel=filters[2], enhanced=enhanced_complexity)

        self.s1_conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.s1_csar4 = CSAR_Block(channel=filters[3], enhanced=enhanced_complexity)

        self.s1_conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.s1_csar5 = CSAR_Block(channel=filters[4], enhanced=enhanced_complexity)

        # Encoder path for Sentinel-2 (5 levels as per paper)
        self.s2_conv1 = conv_block(ch_in=s2_channels, ch_out=filters[0])  # RGB + NIR as per paper
        self.s2_csar1 = CSAR_Block(channel=filters[0], enhanced=enhanced_complexity)

        self.s2_conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.s2_csar2 = CSAR_Block(channel=filters[1], enhanced=enhanced_complexity)

        self.s2_conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.s2_csar3 = CSAR_Block(channel=filters[2], enhanced=enhanced_complexity)

        self.s2_conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.s2_csar4 = CSAR_Block(channel=filters[3], enhanced=enhanced_complexity)

        self.s2_conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.s2_csar5 = CSAR_Block(channel=filters[4], enhanced=enhanced_complexity)

        # Store filter configuration for decoder
        self.filters = filters

        # Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge - combining both paths (dynamic based on filter configuration)
        bridge_channels = self.filters[4] * 2  # Last filter size * 2 (for both paths)
        self.bridge_conv = conv_block(ch_in=bridge_channels, ch_out=bridge_channels)
        self.bridge_csar = CSAR_Block(channel=bridge_channels, enhanced=enhanced_complexity)

        # Decoder path with enhanced CSAR blocks for better feature refinement (dynamic)
        self.up5 = up_conv(ch_in=bridge_channels, ch_out=self.filters[4])
        self.up_conv5 = conv_block(ch_in=self.filters[4]*2, ch_out=self.filters[4])  # After concatenation
        self.up_csar5 = CSAR_Block(channel=self.filters[4], enhanced=enhanced_complexity)

        self.up4 = up_conv(ch_in=self.filters[4], ch_out=self.filters[3])
        self.up_conv4 = conv_block(ch_in=self.filters[3]*2, ch_out=self.filters[3])  # After concatenation
        self.up_csar4 = CSAR_Block(channel=self.filters[3], enhanced=enhanced_complexity)

        self.up3 = up_conv(ch_in=self.filters[3], ch_out=self.filters[2])
        self.up_conv3 = conv_block(ch_in=self.filters[2]*2, ch_out=self.filters[2])  # After concatenation
        self.up_csar3 = CSAR_Block(channel=self.filters[2], enhanced=enhanced_complexity)

        self.up2 = up_conv(ch_in=self.filters[2], ch_out=self.filters[1])
        self.up_conv2 = conv_block(ch_in=self.filters[1]*2, ch_out=self.filters[1])  # After concatenation
        self.up_csar2 = CSAR_Block(channel=self.filters[1], enhanced=enhanced_complexity)

        self.up1 = up_conv(ch_in=self.filters[1], ch_out=self.filters[0])
        self.up_conv1 = conv_block(ch_in=self.filters[0], ch_out=self.filters[0])
        self.up_csar1 = CSAR_Block(channel=self.filters[0], enhanced=enhanced_complexity)

        # Enhanced final output layer with better feature extraction
        self.final_conv = nn.Conv2d(self.filters[0], self.filters[1], kernel_size=3, padding=1, bias=False)
        self.final_norm = nn.BatchNorm2d(self.filters[1])  # Use BatchNorm instead of GroupNorm
        self.final_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.final = nn.Conv2d(self.filters[1], 2, kernel_size=1)

        # Activation functions - choose based on configuration
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        # Concatenation for skip connections
        self.concat = lambda x, y: torch.cat([x, y], dim=1)

    def forward(self, s1_input, s2_input):
        # Sentinel-1 encoding path
        s1_1 = self.s1_conv1(s1_input)
        s1_1 = self.s1_csar1(s1_1)
        s1_1_pool = self.maxpool(s1_1)

        s1_2 = self.s1_conv2(s1_1_pool)
        s1_2 = self.s1_csar2(s1_2)
        s1_2_pool = self.maxpool(s1_2)

        s1_3 = self.s1_conv3(s1_2_pool)
        s1_3 = self.s1_csar3(s1_3)
        s1_3_pool = self.maxpool(s1_3)

        s1_4 = self.s1_conv4(s1_3_pool)
        s1_4 = self.s1_csar4(s1_4)
        s1_4_pool = self.maxpool(s1_4)

        s1_5 = self.s1_conv5(s1_4_pool)
        s1_5 = self.s1_csar5(s1_5)

        # Sentinel-2 encoding path
        s2_1 = self.s2_conv1(s2_input)
        s2_1 = self.s2_csar1(s2_1)
        s2_1_pool = self.maxpool(s2_1)

        s2_2 = self.s2_conv2(s2_1_pool)
        s2_2 = self.s2_csar2(s2_2)
        s2_2_pool = self.maxpool(s2_2)

        s2_3 = self.s2_conv3(s2_2_pool)
        s2_3 = self.s2_csar3(s2_3)
        s2_3_pool = self.maxpool(s2_3)

        s2_4 = self.s2_conv4(s2_3_pool)
        s2_4 = self.s2_csar4(s2_4)
        s2_4_pool = self.maxpool(s2_4)

        s2_5 = self.s2_conv5(s2_4_pool)
        s2_5 = self.s2_csar5(s2_5)

        # Combine at bridge
        bridge_input = self.concat(s1_5, s2_5)
        bridge = self.bridge_conv(bridge_input)
        bridge = self.bridge_csar(bridge)

        # Decoder path with skip connections and CSAR blocks
        d5 = self.up5(bridge)
        d5_concat = self.concat(d5, self.concat(s1_4, s2_4))
        d5 = self.up_conv5(d5_concat)
        d5 = self.up_csar5(d5)

        d4 = self.up4(d5)
        d4_concat = self.concat(d4, self.concat(s1_3, s2_3))
        d4 = self.up_conv4(d4_concat)
        d4 = self.up_csar4(d4)

        d3 = self.up3(d4)
        d3_concat = self.concat(d3, self.concat(s1_2, s2_2))
        d3 = self.up_conv3(d3_concat)
        d3 = self.up_csar3(d3)

        d2 = self.up2(d3)
        d2_concat = self.concat(d2, self.concat(s1_1, s2_1))
        d2 = self.up_conv2(d2_concat)
        d2 = self.up_csar2(d2)

        d1 = self.up1(d2)
        d1 = self.up_conv1(d1)
        d1 = self.up_csar1(d1)

        # Enhanced final output with better feature extraction
        out = self.final_conv(d1)
        out = self.final_norm(out)
        out = self.final_act(out)
        out = self.final(out)

        # Ensure output matches input size (256x256) using adaptive pooling
        # This handles any size mismatches due to enhanced complexity
        out = nn.functional.adaptive_avg_pool2d(out, (256, 256))

        # Apply activation based on configuration
        if self.use_log_softmax:
            # Use log_softmax for numerical stability (recommended for NLLLoss)
            output = self.log_softmax(out)
        else:
            # Use regular softmax (as mentioned in paper)
            output = self.softmax(out)

        return output