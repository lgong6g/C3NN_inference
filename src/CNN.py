# ===== CNN with ESCNN (two block types + realization averaging) =====

import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn.gspaces as gspaces
import escnn.nn as enn

class CNNEmbedding(nn.Module):
    """
    Pipeline:
      1) ESCNN conv blocks WITHOUT pooling (num_blocks_nopool)
      2) ESCNN conv blocks WITH pooling (num_blocks)
      3) Adaptive avg to 1x1, flatten channels
      4) Mean over noise realizations (if present)
      5) Small MLP -> bottleneck (optionally L2-normalize)

    Inputs:
      - [B, R, C, H, W]  (R = # noise realizations)
      - [B, C, H, W]     (treated as R=1)
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        ks = int(hparams.filter_size)
        act_name = (hparams.activation or "relu").lower()
        use_bn = bool(hparams.use_batchnorm)

        # ---- Rotation gspace & FieldTypes
        if hparams.group_N == -1:
            self.gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=hparams.bandlimit)  # SO(2)
        else:
            self.gspace = gspaces.rot2dOnR2(N=hparams.group_N)                          # C_N
        ft_in  = enn.FieldType(self.gspace, hparams.in_channels * [self.gspace.trivial_repr])

        # helper to build modules that need the current FieldType
        def make_act(ft):
            # ReLU works on scalar (trivial) fields; for other choices fallback to ReLU
            return enn.ReLU(ft, inplace=True)  # escnn nonlinearity

        def make_bn(ft):
            return enn.InnerBatchNorm(ft) if use_bn else nn.Identity()

        def make_conv(ft_i, out_channels):
            ft_o = enn.FieldType(self.gspace, int(out_channels) * [self.gspace.trivial_repr])
            conv = enn.R2Conv(in_type=ft_i, out_type=ft_o, kernel_size=ks, bias=False)
            return conv, ft_o

        def make_block(ft_i, out_channels, with_pool: bool):
            layers = []
            # conv1 -> BN -> Act
            conv1, ft_1 = make_conv(ft_i, out_channels)
            layers += [conv1, make_bn(ft_1), make_act(ft_1)]
            # conv2 -> BN -> Act
            conv2, ft_2 = make_conv(ft_1, out_channels)
            layers += [conv2, make_bn(ft_2), make_act(ft_2)]
            # optional spatial downsampling
            if with_pool:
                layers += [enn.PointwiseMaxPool2D(ft_2, kernel_size=2, stride=2)]
            return nn.Sequential(*layers), ft_2

        # ---- Build channels across total depth (non-pooled + pooled)
        total_blocks = int(hparams.num_blocks_nopool) + int(hparams.num_blocks)
        assert total_blocks > 0, "At least one block is required."
        channels = [int(hparams.num_filts) * (2 ** i) for i in range(total_blocks)]

        conv_stages = []
        ft_curr = ft_in

        # Stage 1: Non-pooled blocks
        for i in range(int(hparams.num_blocks_nopool)):
            block, ft_curr = make_block(ft_curr, channels[i], with_pool=False)
            conv_stages.append(block)

        # Stage 2: Pooled blocks
        for j in range(int(hparams.num_blocks)):
            idx = int(hparams.num_blocks_nopool) + j
            block, ft_curr = make_block(ft_curr, channels[idx], with_pool=True)
            conv_stages.append(block)

        self.features = nn.Sequential(*conv_stages)

        # Spatial global pooling to 1x1 (ESCNN wrapper around AdaptiveAvgPool2d)
        self.avgpool = enn.PointwiseAdaptiveAvgPool2D(ft_curr, output_size=1)

        # Keep FieldTypes for wrapping/unwrapping
        self.ft_input  = ft_in
        self.ft_output = ft_curr

        # ---- MLP projection head (standard PyTorch after unwrap)
        hidden_cfg   = hparams.fc_hidden
        hidden_sizes = [hidden_cfg] if isinstance(hidden_cfg, int) else list(hidden_cfg)
        drop_p       = float(getattr(hparams, "dropout", 0.0))

        mlp = []
        prev = channels[-1]  # last ESCNN block outputs this many scalar fields
        for h in hidden_sizes:
            mlp += [nn.Linear(prev, int(h)), nn.ReLU(inplace=True)]
            if drop_p > 0:
                mlp += [nn.Dropout(p=drop_p)]
            prev = int(h)
        mlp += [nn.Linear(prev, int(hparams.bottleneck))]
        self.projection = nn.Sequential(*mlp)

    # ----- Conv/pool stack only (ESCNN path), returns torch.Tensor [B, C_out]
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        need_unsqueeze = (x.dim() == 3)  # [C,H,W]
        if need_unsqueeze:
            x = x.unsqueeze(0)

        # Wrap -> ESCNN modules -> unwrap
        gx = enn.GeometricTensor(x, self.ft_input)
        gx = self.features(gx)                # GeometricTensor
        gx = self.avgpool(gx)                 # GeometricTensor with H=W=1
        x  = gx.tensor                        # [B, C_out, 1, 1]
        x  = torch.flatten(x, 1)              # [B, C_out]

        if need_unsqueeze:
            x = x.squeeze(0)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, R, C, H, W = x.shape
            x = x.view(B * R, C, H, W)
            per_real = self.forward_single(x)     # [B*R, C_out]
            per_real = per_real.view(B, R, -1)    # [B, R, C_out]
            summary  = per_real.mean(dim=1)       # [B, C_out]
        elif x.dim() == 4:
            summary = self.forward_single(x)      # [B, C_out]
        else:
            raise ValueError(f"Expected [B,R,C,H,W] or [B,C,H,W], got {tuple(x.shape)}")

        z = self.projection(summary)              # [B, bottleneck]
        if self.hparams.l2_normalize:
            z = F.normalize(z, p=2, dim=1)
        return z


def build_model(hparams) -> nn.Module:
    model = CNNEmbedding(hparams)
    return model.to(hparams.device)


class CNNEmbedding(nn.Module):
    """
    CNN that:
      1) applies a stack of conv blocks WITHOUT pooling (count = num_blocks_nopool),
      2) then conv blocks WITH pooling (count = num_blocks),
      3) runs per-realization processing and averages across realizations,
      4) maps the summary to a `bottleneck` vector via a small MLP.
    Inputs:
      - [B, R, C, H, W]  (R = # noise realizations)
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
        # Build channels across TOTAL depth (non-pooled + pooled)
        total_blocks = int(hparams.num_blocks_nopool) + int(hparams.num_blocks)
        assert total_blocks > 0, "At least one block (pooled or non-pooled) is required."
        channels = [int(hparams.num_filts) * (2 ** i) for i in range(total_blocks)]
        
        def make_conv_block(in_ch, out_ch, with_pool: bool):
            layers = []
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(Act())
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm2d(out_ch))
            layers.append(Act())
            if with_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)
        # Stage 1: Non-pooled blocks
        in_ch = int(hparams.in_channels)
        conv_blocks = []
        for i in range(int(hparams.num_blocks_nopool)):
            out_ch = channels[i]
            conv_blocks.append(make_conv_block(in_ch, out_ch, with_pool=False))
            in_ch = out_ch
        # Stage 2: Pooled blocks
        for j in range(int(hparams.num_blocks)):
            idx = int(hparams.num_blocks_nopool) + j
            out_ch = channels[idx]
            conv_blocks.append(make_conv_block(in_ch, out_ch, with_pool=True))
            in_ch = out_ch
        self.features = nn.Sequential(*conv_blocks)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        # Projection head (MLP -> bottleneck)
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
    
    # Convs + pool summary (no FC) — can accept [B,C,H,W] or [C,H,W]
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
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
        if x.dim() == 5:
            B, R, C, H, W = x.shape
            x = x.view(B * R, C, H, W)
            per_real = self.forward_single(x)      # [B*R, C_out]
            per_real = per_real.view(B, R, -1)     # [B, R, C_out]
            summary  = per_real.mean(dim=1)        # [B, C_out]
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
