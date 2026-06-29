import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    """
    Simple model that matches the saved model architecture
    Based on the keys found in the saved model state_dict
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        # Sentinel-1 branch (single channel input)
        self.s1_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.s1_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.s1_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Sentinel-2 branch (8 channel input based on saved model)
        self.s2_conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.s2_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.s2_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # S2 channel reduction (Conv2d based on saved model)
        self.s2_reduce = nn.Conv2d(12, 8, kernel_size=1)

        # Attention mechanism (Conv2d based on saved model)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),  # 128 + 128 = 256 from concatenated features
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoder layers (Conv2d based on saved model)
        self.decoder1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Classifier (2 classes based on saved model)
        self.classifier = nn.Conv2d(32, 2, kernel_size=1)

        # Batch normalization layers (matching saved model dimensions)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, s1, s2):
        batch_size = s1.size(0)

        # Reduce S2 channels from 12 to 8
        s2_reduced = self.s2_reduce(s2)

        # Process Sentinel-1
        s1_feat = F.relu(self.s1_conv1(s1))
        s1_feat = F.max_pool2d(s1_feat, 2)
        s1_feat = F.relu(self.s1_conv2(s1_feat))
        s1_feat = F.max_pool2d(s1_feat, 2)
        s1_feat = F.relu(self.s1_conv3(s1_feat))  # [batch_size, 128, H/4, W/4]

        # Process Sentinel-2
        s2_feat = F.relu(self.s2_conv1(s2_reduced))
        s2_feat = F.max_pool2d(s2_feat, 2)
        s2_feat = F.relu(self.s2_conv2(s2_feat))
        s2_feat = F.max_pool2d(s2_feat, 2)
        s2_feat = F.relu(self.s2_conv3(s2_feat))  # [batch_size, 128, H/4, W/4]

        # Concatenate features
        combined_feat = torch.cat([s1_feat, s2_feat], dim=1)  # [batch_size, 256, H/4, W/4]

        # Apply attention
        attention_weights = self.attention(combined_feat)
        attended_feat = combined_feat * attention_weights

        # Decode
        x = F.relu(self.decoder1(attended_feat))  # [batch_size, 128, H/4, W/4]
        x = self.bn3(x)  # bn3 is for 128 channels
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.relu(self.decoder2(x))  # [batch_size, 64, H/2, W/2]
        x = self.bn2(x)  # bn2 is for 64 channels
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.relu(self.decoder3(x))  # [batch_size, 32, H, W]
        x = self.bn1(x)  # bn1 is for 32 channels

        # Classify
        output = self.classifier(x)  # [batch_size, 2, H, W]

        # Apply softmax and take the burned class (index 1)
        output = F.softmax(output, dim=1)
        output = output[:, 1:2, :, :]  # Take only the burned class, keep channel dimension

        return output
