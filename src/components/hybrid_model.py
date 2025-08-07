import torch
import torch.nn as nn

# Updated SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Updated Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return F.relu(x)

# Visual Backbone
class ResEmoteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # (B, 64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (B, 64, 56, 56)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (B, 128, 28, 28)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (B, 256, 14, 14)
        )
        self.se = SEBlock(256)
        self.res1 = ResidualBlock(256, 512, 2)  # (B, 512, 7, 7)
        self.res2 = ResidualBlock(512, 1024, 2)  # (B, 1024, 4, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = 1024

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).view(x.size(0), -1)
        return x

# Landmark Branch with optional reconstruction head
class LandmarkBranch(nn.Module):
    def __init__(self, output_reconstruction=True):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1404, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_reconstruction = output_reconstruction
        if output_reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1404)  # Match input dimension for reconstruction
            )

    def forward(self, x):
        encoded = self.encoder(x)
        if self.output_reconstruction:
            decoded = self.decoder(encoded)
            return encoded, decoded
        else:
            return encoded, None


# Cross-Modality Fusion (optional)
class FusionModule(nn.Module):
    def __init__(self, img_dim, lm_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim + lm_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, img_feat, lm_feat):
        x = torch.cat([img_feat, lm_feat], dim=1)
        return self.fc(x)


# Hybrid Model with projection heads for contrastive loss
class HybridEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = ResEmoteNet()
        self.landmarks = LandmarkBranch(output_reconstruction=True)
        self.fusion = FusionModule(self.visual.out_features, 256)
        self.classifier = nn.Linear(128, 7)

        # Add projection heads for contrastive loss
        self.projection_img = nn.Linear(1024, 128)
        self.projection_lm = nn.Linear(256, 128)

    def forward(self, img, lm):
        img_feat = self.visual(img)                    # (B, 1024)
        lm_feat, lm_recon = self.landmarks(lm)         # (B, 256), (B, 936)
        fused = self.fusion(img_feat, lm_feat)         # (B, 128)
        logits = self.classifier(fused)                # (B, 7)

        # Project to same space for cosine contrastive loss
        img_proj = self.projection_img(img_feat)       # (B, 128)
        lm_proj = self.projection_lm(lm_feat)          # (B, 128)

        return logits, img_proj, lm_proj, lm_recon