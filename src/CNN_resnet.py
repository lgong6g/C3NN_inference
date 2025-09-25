import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


# CNN embedding + final regression head
class CNNResEmbedding(nn.Module):
    def __init__(self, n_block=10, n_ch=64, embed_dim=128, out_dim=8):
        """
        n_block: number of residual blocks
        n_ch: base number of channels
        embed_dim: hidden embedding dimension
        out_dim: final output dimension (e.g., number of cosmological parameters)
        """
        super().__init__()

        # Initial convolution (input has 4 channels)
        self.conv1 = nn.Conv2d(4, n_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(n_ch)

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(n_ch) for _ in range(n_block)])

        # Downsampling stages (adapted to 50x50 input)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_ch, 2*n_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(2*n_ch)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(2*n_ch, 4*n_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(4*n_ch)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(4*n_ch, 8*n_ch, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(8*n_ch)

        # Final pooling to collapse to 1x1
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected head
        self.fc1 = nn.Linear(4*n_ch, embed_dim)
        self.fc2 = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        # x: (batch, n_samples, channels, H, W) = (1250, 150, 4, 50, 50)
        B, S, C, H, W = x.shape

        # Merge batch and samples for CNN
        x = x.view(B*S, C, H, W)  # (1250*150, 4, 50, 50)

        out = F.relu(self.bn1(self.conv1(x))) # (1250*150, 4, 25, 25)
        out = self.res_blocks(out)

        out = self.pool1(out) # (1250*150, 4, 12, 12)
        out = F.relu(self.bn2(self.conv2(out))) # (1250*150, 4, 10, 10)
        out = self.pool2(out) # (1250*150, 4, 5, 5)
        out = F.relu(self.bn3(self.conv3(out))) # (1250*150, 4, 3, 3)
        out = self.pool4(out) # (1250*150, 4, 1, 1)
        #out = F.relu(self.bn4(self.conv4(out)))
        #out = self.pool4(out)                   # -> (B*S, 8*n_ch, 1, 1)

        out = torch.flatten(out, 1)             # (B*S, 8*n_ch)
        out = F.relu(self.fc1(out))             # (B*S, 256)
        out = self.fc2(out)                     # (B*S, out_dim)

        # Reshape back: group per cosmology
        out = out.view(B, S, -1)                # (1250, 150, out_dim)

        # Aggregate predictions over samples
        out = out.mean(dim=1)                   # (1250, out_dim)

        return out