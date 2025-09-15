# ---------------- CNN Model ---------------- #
# ---------------- CNN with Realization Averaging ---------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmbedding(nn.Module):
    """
    CNN that:
      1) runs the conv/pool stack on each noise realization,
      2) averages the per-realization summaries,
      3) passes through a small MLP head to produce a length-`bottleneck` vector.

    Compatible inputs:
      - [B, R, C, H, W]  (B=batch, R=realizations, C=channels)
      - [B, C, H, W]     (treated as R=1)
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        ks       = int(hparams.filter_size)
        padding  = ks // 2
        use_bn   = bool(hparams.use_batchnorm)
        act_name = (hparams.activation or "relu").lower()
        Act      = nn.ReLU if act_name == "relu" else nn.GELU

        # channels per block: base * 2**i
        channels = [int(hparams.num_filts) * (2 ** i) for i in range(int(hparams.num_blocks))]

        in_ch = int(hparams.in_channels)
        conv_blocks = []
        for out_ch in channels:
            block = []
            block.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn))
            if use_bn: block.append(nn.BatchNorm2d(out_ch))
            block.append(Act())
            block.append(nn.Conv2d(out_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn))
            if use_bn: block.append(nn.BatchNorm2d(out_ch))
            block.append(Act())
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            conv_blocks.append(nn.Sequential(*block))
            in_ch = out_ch

        self.features = nn.Sequential(*conv_blocks)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))

        # Small projection head (your fc1/relu -> fc2/relu -> fc3 analogue)
        hidden_cfg   = hparams.fc_hidden
        hidden_sizes = [hidden_cfg] if isinstance(hidden_cfg, int) else list(hidden_cfg)
        drop_p       = float(getattr(hparams, "dropout", 0.0))

        mlp = []
        prev = channels[-1]
        for h in hidden_sizes:
            mlp.append(nn.Linear(prev, int(h)))
            mlp.append(Act())
            if drop_p > 0:
                mlp.append(nn.Dropout(p=drop_p))
            prev = int(h)
        mlp.append(nn.Linear(prev, int(hparams.bottleneck)))
        self.projection = nn.Sequential(*mlp)

        self.apply(self._init_weights)

    # ---- Your "forward_single" analogue: convs + pool, no FC ----
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] or [C, H, W]
        returns: [B, C_out] or [C_out]
        """
        need_unsqueeze = (x.dim() == 3)  # [C,H,W]
        if need_unsqueeze:
            x = x.unsqueeze(0)
        x = self.features(x)         # [B, C_out, H', W']
        x = self.avgpool(x)          # [B, C_out, 1, 1]
        x = torch.flatten(x, 1)      # [B, C_out]
        if need_unsqueeze:
            x = x.squeeze(0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If x is [B, R, C, H, W]:
          - Run forward_single on all B*R images (shared weights)
          - Reshape to [B, R, C_out] and mean over R
          - Project to bottleneck
        If x is [B, C, H, W], treat R=1.
        """
        if x.dim() == 5:
            B, R, C, H, W = x.shape
            x = x.view(B * R, C, H, W)
            per_real = self.forward_single(x)      # [B*R, C_out]
            per_real = per_real.view(B, R, -1)     # [B, R, C_out]
            summary  = per_real.mean(dim=1)        # [B, C_out]  (mean over realizations)
        elif x.dim() == 4:
            summary = self.forward_single(x)       # [B, C_out]
        else:
            raise ValueError(
                f"Expected input of shape [B,R,C,H,W] or [B,C,H,W], got {tuple(x.shape)}"
            )

        z = self.projection(summary)                # [B, bottleneck]
        if self.hparams.l2_normalize:
            z = F.normalize(z, p=2, dim=1)
        return z

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

