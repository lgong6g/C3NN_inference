"""
C3NN compression network with Mixed Density Network (MDN) head,
trained via Variational Mutual Information Maximization (VMIM).

Stage 1 of the two-stage NLE pipeline:
  - C3NN_hybrid_highorder backbone:
      * Takes 5-channel input [B, n_maps, 5, H, W] (4 convergence + 1 2PCF channel).
      * Passes the 4 convergence channels through NonlinearConv to compute
        correlators C1 ... C_order.
      * Discards C1 and C2 (2-point information already captured by the external
        2PCF measurement), retaining only C3 and higher-order correlators.
      * Concatenates the mean of C3+ features with the compressed 2PCF vector.
  - A learnable FC layer + nonlinear activation (GELU) compresses the backbone
    output to `compressed_dim`.
  - An MDN head predicts q(theta | T(x)) as a K-component Gaussian mixture.
  - Training minimises the MDN NLL:
        L = -E [ log sum_k pi_k * N(theta ; mu_k, sigma_k) ]
    which is a Monte-Carlo estimate of the negative VMIM lower bound
        I_VMIM(T(x); theta) >= E[ log q_MDN(theta | T(x)) ] + const
    Maximising this bound encourages T(x) to be maximally informative
    about the cosmological parameters theta.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from C3NN_extensions_modified import NonlinearConv, Mean2d


def compute_valid_masks(
    representative_maps: torch.Tensor,
    filter_size: int,
    threshold: float = 0.90,
) -> torch.Tensor:
    """Per-cutout boolean masks of positions where ≥threshold of the filter footprint is unmasked.

    Each of the n_maps cutouts has a distinct survey footprint that is fixed across
    all cosmologies.  This function computes one mask per cutout using a uniform
    box-filter convolution over the binary (non-zero) pixel map.

    Args:
        representative_maps: [n_maps, H, W] tensor from the first cosmology (any channel);
                             zero pixels are treated as masked.
        filter_size: C3NN spatial filter size (same kernel size as NonlinearConv).
        threshold: minimum unmasked fraction required (default 0.90).

    Returns:
        [n_maps, H, W] bool tensor — True at every valid position for each cutout.
    """
    n_maps, H, W = representative_maps.shape
    binary = (representative_maps != 0).float().unsqueeze(1)  # [n_maps, 1, H, W]
    pad = (filter_size - 1) // 2
    uniform = torch.ones(1, 1, filter_size, filter_size,
                         device=representative_maps.device) / (filter_size * filter_size)
    # Process all maps in one batched conv call
    unmasked_frac = F.conv2d(binary, uniform, padding=pad)[:, 0]  # [n_maps, H, W]
    return unmasked_frac >= threshold


# =============================================================================
# Hybrid high-order backbone
# =============================================================================

class C3NN_hybrid_highorder(nn.Module):
    """C3NN hybrid backbone: C3-and-higher correlators concatenated with 2PCF.

    Input shape:  [B, n_maps, 5, H, W]
        Channels 0-3 : convergence maps (4 tomographic redshift bins).
        Channel  4   : compressed 2PCF padded to the same H x W grid.

    Processing:
        1. Extract the 2PCF vector from the last channel (first n_2pcf pixels
           of the first pixel row, averaged over map realisations).
        2. Pass the 4 convergence channels through NonlinearConv, producing
           correlator maps C1 ... C_order for each filter.
        3. Average spatially (Mean2d) then average over realisations.
        4. Discard C1 (indices 0 : num_filts) and C2 (indices num_filts : 2*num_filts),
           keeping only C3 and above (indices 2*num_filts : order*num_filts).
        5. Concatenate C3+ mean features with the 2PCF vector.

    Output shape: [B, (order - 2) * num_filts + n_2pcf]
        e.g. for order=4, num_filts=40, n_2pcf=8 → [B, 88]
    """

    def __init__(self, hparams, n_2pcf: int = 17):
        """
        Args:
            hparams:  Hyperparameter container.  Uses in_channels (4),
                      num_filts, filter_size, order.
            n_2pcf:   Number of compressed 2PCF features stored in the
                      last input channel (default 8).
        """
        super().__init__()
        self.in_channels      = hparams.in_channels   # 4 convergence channels
        self.num_filts        = hparams.num_filts
        self.filter_size      = hparams.filter_size
        self.order            = hparams.order
        self.n_2pcf           = n_2pcf

        # NonlinearConv outputs [C1, C2, C3, ..., C_order] along channel dim.
        # C1: [0 : num_filts], C2: [num_filts : 2*num_filts], C3+: [2*num_filts :]
        self.highorder_start = 2 * hparams.num_filts
        self.highorder_dim   = (hparams.order - 2) * hparams.num_filts  # C3+ only
        self.output_dim      = self.highorder_dim + n_2pcf

        # C3NN correlator convolution (rotational equivariance via escnn)
        self.corr_conv   = NonlinearConv(self.in_channels, self.num_filts,
                                         self.filter_size, self.order)
        self.spatial_mean = Mean2d()

        # Per-cutout valid-position masks: [n_maps, H, W] bool.
        # Set via set_valid_masks() before training; None falls back to full spatial mean.
        self.register_buffer('valid_masks', None)

        # BatchNorm over C3+ features only.
        # Replaced by nn.Identity() inside C3NN_MDN_Compressor to avoid
        # double-normalisation (we apply our own BN after the FC layer).
        self.bn = nn.BatchNorm1d(
            self.highorder_dim, affine=False, eps=1e-5, dtype=torch.float32
        )

        # vmap over the batch dimension for efficient per-realization encoding
        #self.parallel = torch.vmap(self._forward_single, in_dims=0, out_dims=0)

        # Deferred BN flag — mirrors pattern in existing C3NN modules
        self._bn_ready = False

    # ------------------------------------------------------------------
    def set_valid_masks(self, masks: torch.Tensor) -> None:
        """Register per-cutout valid-position masks.

        Args:
            masks: [n_maps, H, W] bool tensor produced by compute_valid_masks().
                   Must be on the same device as the model.
        """
        self.valid_masks = masks.bool()

    # ------------------------------------------------------------------
    def _forward_single(self, x):
        """Encode one cosmology's maps: NonlinearConv -> masked spatial mean.

        Each map uses its own precomputed valid-position mask (its cutout's
        survey footprint).  Positions where <90% of the filter footprint is
        unmasked are excluded from the spatial average.

        Args:
            x: Tensor [n_maps, 4, H, W]

        Returns:
            Tensor [n_maps, order * num_filts]
        """
        x = self.corr_conv(x)    # [n_maps, order*num_filts, H, W]
        if self.valid_masks is not None:
            # Each map m has its own footprint; average only over its valid positions.
            # x[m, :, valid_masks[m]] -> [order*num_filts, n_valid_m] -> mean -> [order*num_filts]
            x = torch.stack([
                x[m, :, self.valid_masks[m]].mean(dim=-1)
                for m in range(x.shape[0])
            ])  # [n_maps, order*num_filts]
        else:
            x = self.spatial_mean(x)  # fallback: full spatial mean
        return x

    # ------------------------------------------------------------------
    def forward(self, x, chop: bool = False):
        """Encode a batch of 5-channel simulation data.

        Args:
            x:    Tensor [B, n_maps, 5, H, W]
            chop: If True, skip BatchNorm (used internally by sbi for
                  single-sample calls).

        Returns:
            Tensor [B, (order - 2) * num_filts + n_2pcf]
        """
        # --- 2PCF: last input channel, averaged over realisations ------
        # x[:, :, -1] -> [B, n_maps, H, W]
        # Mean over realizations -> [B, H, W], then take first row, first n_2pcf pixels
        compressed_2pt = x[:, :, -1].mean(dim=1)[:, 0, :self.n_2pcf]  # [B, n_2pcf]

        # --- C3NN on 4 convergence channels ----------------------------
        #maps = x[:, :, :-1]                  # [B, n_maps, 4, H, W]
        #summaries = self.parallel(maps)       # [B, n_maps, order*num_filts]

        maps = x[:, :, :-1]                  # [B, n_maps, 4, H, W]
        B, n_maps, C, H, W = maps.shape

        # Flatten B and n_maps into a single batch dimension so NonlinearConv
        # runs ONE conv2d call on [B*n_maps, 4, H, W] instead of B sequential
        # calls on [n_maps, 4, H, W].  This removes the Python loop overhead
        # and lets the GPU process all cosmologies in parallel.
        #
        # Memory: with c3nn_batch_size=8 and n_maps=3, the flat batch is
        # [24, 4, 512, 512].  Peak backward intermediates = 13 × [24, 40, 512, 512]
        # ≈ 13.1 GB, well within the 40 GB A100 budget.  Gradient checkpointing
        # is therefore not needed and its sequential serialisation is removed.
        maps_flat = maps.reshape(B * n_maps, C, H, W)           # [B*n_maps, C, H, W]
        out_flat  = self.corr_conv(maps_flat)                    # [B*n_maps, order*num_filts, H, W]
        out       = out_flat.reshape(B, n_maps, -1, H, W)        # [B, n_maps, order*num_filts, H, W]

        if self.valid_masks is not None:
            # Iterate over maps (n_maps iterations, all B cosmologies in parallel).
            # This replaces the old B*n_maps-iteration pattern from _forward_single.
            summaries = torch.stack(
                [out[:, m, :, self.valid_masks[m]].mean(dim=-1) for m in range(n_maps)],
                dim=1,
            )  # [B, n_maps, order*num_filts]
        else:
            summaries = out.mean(dim=(-2, -1))  # [B, n_maps, order*num_filts]
        
        # Average correlator features over realisations
        mean = summaries.mean(dim=1)          # [B, order*num_filts]

        # Discard C1 and C2 — keep C3 and higher
        highorder = mean[:, self.highorder_start:]  # [B, highorder_dim]

        if self._bn_ready and not chop:
            highorder = self.bn(highorder)
        self._bn_ready = True

        # Concatenate C3+ summary with external 2PCF measurement
        return torch.cat((highorder, compressed_2pt), dim=1)  # [B, output_dim]


# =============================================================================
# MDN compressor
# =============================================================================

class C3NN_MDN_Compressor(nn.Module):
    """C3NN_hybrid_highorder backbone + FC + activation + MDN head for VMIM.

    Architecture (per batch of cosmologies):
        x  [B, n_maps, 5, H, W]
            --> C3NN_hybrid_highorder  --> [B, raw_dim]          (C3+ ‖ 2PCF)
            --> Linear + GELU + BN     --> [B, compressed_dim]   (learned T(x))
            --> MDN head               --> (pi, mu, log_sigma) over theta

    compress() returns T(x) — this is the fixed input to Stage 2 NLE once
    all Stage-1 weights are frozen.
    """
    """
    def __init__(
        self,
        backbone: nn.Module,
        raw_dim: int,
        compressed_dim: int,
        n_params: int,
        n_components: int = 8,
        activation: nn.Module = None,
    ):
    """
    def __init__(
      self,
      backbone: nn.Module,
      raw_dim: int,          # now 
      compressed_dim: int,
      n_params: int,
      n_components: int = 1,
      activation: nn.Module = None,
      n_2pcf: int = 17,     
    ):
        """
        Args:
            backbone:        C3NN_hybrid_highorder (or compatible) module mapping
                             x [B, n_maps, 5, H, W] -> [B, raw_dim].
                             Its internal BatchNorm is replaced by Identity.
            raw_dim:         Output dimensionality of the backbone
                             (= (order-2)*num_filts + n_2pcf).
                             NEW: (= (order-2)*num_filts) only. The n_2pcf features are appended                              after fc_compress, not compressed.
            compressed_dim:  Target summary dimension after FC compression
                             (= hparams.out_dim, e.g. 20).
            n_params:        Dimensionality of the cosmological parameter space
                             (e.g. 3 for Om, sigma8, w0).
            n_components:    Number of Gaussian mixture components K.
            activation:      Nonlinear activation applied after fc_compress.
                             Defaults to nn.GELU() when None.
            n_2pcf:          NEW: compressed 2PCF dimension, appended after fc_compress
        """
        super().__init__()
        self.n_components  = n_components
        self.n_params      = n_params
        self.compressed_dim = compressed_dim

        # Disable backbone's internal BN to avoid double-normalisation.
        backbone.bn = nn.Identity()
        self.backbone = backbone

        # FC compression: Linear -> nonlinear activation -> BN
        self.fc_compress = nn.Linear(raw_dim, compressed_dim)
        self.activation  = activation if activation is not None else nn.GELU()
        self.bn_compress = nn.BatchNorm1d(
            compressed_dim, affine=False, eps=1e-5, dtype=torch.float32
        )

        # MDN head: pi logits + mu + log_sigma for K components over R^n_params
        mdn_out_dim = n_components * (1 + 2 * n_params)
        # self.mdn_head = nn.Linear(compressed_dim, mdn_out_dim)
        self.n_2pcf   = n_2pcf
        self.mdn_head = nn.Linear(compressed_dim + n_2pcf, mdn_out_dim)

        # Deferred BN: skip on the very first call (batch size may be 1 in sbi)
        self._bn_ready = False

    # ------------------------------------------------------------------
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return compressed summary T(x) without MDN output.

        Called during Stage 2 to extract fixed summaries from the frozen
        Stage-1 compressor for all training simulations.

        Args:
            x: Tensor [B, n_maps, 5, H, W]

        Returns:
            Tensor [B, compressed_dim]
        NEW: 
        Returns:
            Tensor [B, compressed_dim + n_2pcf]
            First compressed_dim entries: FC-compressed high-order summary.
            Last n_2pcf entries: raw compressed 2PCF, passed through unchanged.
        """
        
        """
        features   = self.backbone(x)                       # [B, raw_dim]
        compressed = self.activation(self.fc_compress(features))  # [B, compressed_dim]
        if self._bn_ready:
            compressed = self.bn_compress(compressed)
        self._bn_ready = True
        return compressed
        """
        
        features   = self.backbone(x)                              # [B, highorder_dim + n_2pcf]
        #features = self.backbone.forward(x)
        highorder  = features[:, :self.fc_compress.in_features]    # [B, highorder_dim]
        twopcf     = features[:, self.fc_compress.in_features:]    # [B, n_2pcf]
        compressed = self.activation(self.fc_compress(highorder))  # [B, compressed_dim]
        if self._bn_ready:
            compressed = self.bn_compress(compressed)
        self._bn_ready = True
        return torch.cat([compressed, twopcf], dim=1)              # [B, compressed_dim + n_2pcf]
        
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Full forward: map batch -> MDN parameters over theta.

        Args:
            x: Tensor [B, n_maps, 5, H, W]

        Returns:
            pi_logits: Tensor [B, K]           -- raw (pre-softmax) mixing weights
            mu:        Tensor [B, K, n_params]  -- component means
            log_sigma: Tensor [B, K, n_params]  -- log component standard deviations
        """
        compressed = self.compress(x)         # [B, compressed_dim]
        mdn_out    = self.mdn_head(compressed) # [B, K*(1 + 2*n_params)]

        K, D = self.n_components, self.n_params
        pi_logits = mdn_out[:, :K]                             # [B, K]
        mu        = mdn_out[:, K : K + K * D].view(-1, K, D)  # [B, K, D]
        log_sigma = mdn_out[:, K + K * D :].view(-1, K, D)    # [B, K, D]

        return pi_logits, mu, log_sigma


# =============================================================================
# VMIM loss function
# =============================================================================

def mdn_nll_loss(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """MDN negative log-likelihood — the VMIM compression loss.

    Computes:
        L = -E_batch [ log sum_k pi_k * N(theta ; mu_k, sigma_k) ]

    Minimising L maximises E[log q_MDN(theta | T(x))], a variational
    lower bound on the mutual information I(T(x); theta).

    Args:
        pi_logits: Tensor [B, K]    -- raw (pre-softmax) mixing weights
        mu:        Tensor [B, K, D] -- component means
        log_sigma: Tensor [B, K, D] -- log component standard deviations
        theta:     Tensor [B, D]    -- ground-truth cosmological parameters

    Returns:
        Scalar mean NLL over the batch.
    """
    # Promote to float32 so that exp/division/logsumexp are not squeezed
    # into float16's narrow exponent range under autocast.
    pi_logits = pi_logits.float()
    mu        = mu.float()
    log_sigma = log_sigma.float()
    theta     = theta.float()

    # Clamp log_sigma to prevent sigma from underflowing to 0 (MDN collapse).
    # sigma in [exp(-7), exp(7)] ≈ [9e-4, 1097] is ample for normalised params.
    log_sigma = log_sigma.clamp(min=-7.0, max=7.0)

    log_pi = F.log_softmax(pi_logits, dim=-1)     # [B, K]
    sigma  = torch.exp(log_sigma)                 # [B, K, D]

    # Expand theta: [B, 1, D] -> broadcast [B, K, D]
    diff = (theta.unsqueeze(1) - mu) / sigma      # [B, K, D]

    # Per-component Gaussian log-density, summed over parameter dims
    log_gauss = -0.5 * (
        diff.pow(2) + 2.0 * log_sigma + math.log(2.0 * math.pi)
    ).sum(dim=-1)                                 # [B, K]

    # Log mixture density via numerically stable log-sum-exp
    log_mix = torch.logsumexp(log_pi + log_gauss, dim=-1)  # [B]

    return -log_mix.mean()
