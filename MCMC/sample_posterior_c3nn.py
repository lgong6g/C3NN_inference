#!/usr/bin/env python
"""sample_posterior_c3nn.py — GPU posterior sampling for the two-stage C3NN NLE.

Loads the Stage-2 NLE network (cloudpickle .pkl) trained in
train_nle.py, constructs a PolytopeBoxUniform prior from the
training-parameter file, and samples the posterior p(θ | x_obs) with GPU-
accelerated affine-invariant ensemble MCMC (Goodman & Weare stretch move).

This directly replicates the algorithm in NLE.get_posterior_gpu from
MCMC/ml_methods.py, adapted to use the nflows-based NLE network whose
log_prob signature is:

    nle_net.log_prob(x, condition=theta)

where x has shape (1, n_active, data_dim) and theta has shape (n_active, ndim),
returning shape (1, n_active).  This matches the training convention in
train_nle.py:

    val_lp = nle_net.log_prob(x_nle_val.unsqueeze(0), condition=t_nle_val)[0]

Two compression modes are supported via --compression-mode:

  joint (default)
    The Stage-1 C3NN backbone computes C3+ higher-order correlators and reads
    the MOPED 2PCF coefficients from the 5th input channel.  Both are passed
    together through the FC compression layer, producing a single fixed-length
    summary vector of shape (compressed_dim,).  This is the original pipeline
    produced by the old training runs (compressor pkl whose fc_compress input
    size equals highorder_dim + n_2pcf).

  separate
    The FC compression layer operates only on the higher-order correlators,
    producing a (compressed_dim,) vector.  The MOPED 2PCF coefficients are then
    concatenated to yield a (compressed_dim + n_2pcf,) summary.  This matches
    exactly how the C3NN and NLE are trained in train_c3nn_ddp.py and
    train_nle.py (output files carry the suffix _moped_separate_).

In both modes the C3NN_MDN_Compressor.compress() method adapts automatically
based on the loaded compressor's fc_compress.in_features, so no changes to the
compressor code are needed.

The observation (x_obs) can be supplied in three ways:

  --x-obs <path.npy>      Load a pre-computed summary from disk.
  --x-obs-index <int>     Use a row from the saved training summaries
                          (handy for validation / closure tests).
  --maps / --twopcf       Compress raw maps on the fly (requires --compressor).

Default model paths are set automatically from --compression-mode; override
any path individually with the explicit flags below.

Default model paths point to the trained model at:
  /pscratch/sd/l/lgong/C3NN_inference/C3NN_Models/
  order_4_filters_20_size_37_lr_1.00e-03_features_64_blocks_5_transforms_5_batchsize_32_*

Usage examples
--------------
# Closure test with joint compression (default mode)
python sample_posterior_c3nn.py --x-obs-index 0 --output test_closure.npy

# Closure test with separate compression (matches train_c3nn_ddp.py)
python sample_posterior_c3nn.py --compression-mode separate \\
    --x-obs-index 0 --output test_closure_sep.npy

# Real inference from a pre-computed summary
python sample_posterior_c3nn.py \\
    --x-obs /path/to/obs_summary.npy \\
    --output posterior_samples.npy \\
    --nwalkers 1000 --nsteps 2000 --discard 5000 --thin 3

# On-the-fly map compression (separate mode)
python sample_posterior_c3nn.py --compression-mode separate \\
    --compressor /path/to/*_moped_separate_compressor.pkl \\
    --maps /path/to/obs_maps.npy \\
    --twopcf /path/to/obs_2pcf.npy \\
    --output posterior_samples.npy

Activate environment first:
    source /global/common/software/m1727/lgong/sbi_env/bin/activate
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path for compress_observation (C3NN_MDN_compressor import)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import cloudpickle as cpk
from typing import List, Tuple

from sbi.utils import BoxUniform

try:
    from tqdm import trange
except ImportError:
    trange = None


# ---------------------------------------------------------------------------
# PolytopeBoxUniform — inlined from MCMC/utils.py (avoids the optuna import)
# ---------------------------------------------------------------------------

class PolytopeBoxUniform(BoxUniform):
    """BoxUniform restricted to convex polygon constraints on parameter subsets.

    Identical to the class in MCMC/utils.py; inlined here so the script does
    not require the full utils.py (which imports optuna, unavailable in sbi_env).
    """

    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        polygon_constraints: List[Tuple[Tuple[int, ...], torch.Tensor]],
        device: str = "cpu",
    ) -> None:
        super().__init__(low=low, high=high, device=device)
        self._device = device
        self._planes: List[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = []
        for dims, verts in polygon_constraints:
            verts = verts.to(device).float()
            normals, offsets = self._half_planes(verts)
            self._planes.append((tuple(dims), normals, offsets))

    @staticmethod
    def _half_planes(
        vertices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edges   = torch.roll(vertices, -1, 0) - vertices          # (M, 2)
        normals = torch.stack([edges[:, 1], -edges[:, 0]], dim=1) # (M, 2)
        centroid = vertices.mean(0)
        outward  = (normals * (vertices - centroid)).sum(-1) > 0  # (M,)
        normals  = torch.where(outward.unsqueeze(1), normals, -normals)
        offsets  = (normals * vertices).sum(-1)                   # (M,)
        return normals, offsets

    def _in_polytopes(self, theta: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(theta.shape[:-1], dtype=torch.bool, device=theta.device)
        for dims, normals, offsets in self._planes:
            pts    = theta[..., list(dims)]
            margin = torch.einsum("md,...d->...m", normals, pts) - offsets
            mask   = mask & (margin <= 0).all(-1)
        return mask

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        base = super().log_prob(theta)
        return torch.where(
            self._in_polytopes(theta),
            base,
            torch.full_like(base, -float("inf")),
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        n = int(sample_shape.numel())
        out: List[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            cand = super().sample((max(remaining * 4, 1),))
            kept = cand[self._in_polytopes(cand)][:remaining]
            out.append(kept)
            remaining -= kept.shape[0]
        result = torch.cat(out, dim=0)[:n]
        return result.reshape(*sample_shape, self.event_shape[0])


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_MODEL_DIR = "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models"
_TAG = (
    "order_4_filters_20_size_37_lr_1.00e-03_"
    "features_64_blocks_5_transforms_5_batchsize_32"
)

# joint mode: FC layer compresses (higher-order + 2PCF) jointly → (compressed_dim,)
_DEFAULTS = {
    "joint": {
        "nle_net":       f"{_MODEL_DIR}/{_TAG}_nle_stage2_nle_net.pkl",
        "compressor":    f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_compressor.pkl",
        "nle_params":    f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_nle_params.npy",
        "nle_summaries": f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_nle_summaries.npy",
    },
    # separate mode: FC layer compresses only higher-order → (compressed_dim,),
    # then 2PCF is concatenated → (compressed_dim + n_2pcf,).
    # Matches train_c3nn_ddp.py + train_nle.py (files carry _moped_separate_ suffix).
    "separate": {
        "nle_net":       f"{_MODEL_DIR}/{_TAG}_nle_stage2_nle_moped_separate_5coeffs_net.pkl",
        "compressor":    f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_moped_separate_5coeffs_compressor.pkl",
        "nle_params":    f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_nle_moped_separate_5coeffs_params.npy",
        "nle_summaries": f"{_MODEL_DIR}/{_TAG}_c3nn_stage1_nle_moped_separate_5coeffs_summaries.npy",
    },
}

# Parameter names — first 5 columns of full_parameters_2500nodes_Om_sigma8_w0_Aia_Mc_nu.npy
PARAM_NAMES = ["Om", "sigma8", "w0", "A_ia", "bary_Mc"]


# ---------------------------------------------------------------------------
# Prior construction
# ---------------------------------------------------------------------------

def build_prior(theta: np.ndarray, device: str) -> PolytopeBoxUniform:
    """PolytopeBoxUniform prior from the min/max of training parameters.

    Applies the same (Om, sigma8) and (Om, w0) polygon cuts used in the
    MOPED-NLE analysis (MCMC/ml_methods.py NDE.__init__).
    """
    theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
    low  = theta_t.min(dim=0).values
    high = theta_t.max(dim=0).values

    polygon_constraints = []

    om_ind   = PARAM_NAMES.index("Om")
    sig8_ind = PARAM_NAMES.index("sigma8")
    w0_ind   = PARAM_NAMES.index("w0")

    poly_om_sig8 = torch.tensor([
        [0.1, 1.4], [0.2, 1.4], [0.5, 0.65],
        [0.5, 0.4], [0.3, 0.4], [0.1, 0.9],
    ], device=device, dtype=torch.float32)
    polygon_constraints.append(((om_ind, sig8_ind), poly_om_sig8))

    poly_om_w0 = torch.tensor([
        [0.1, -1.1], [0.5, -2.0], [0.5, -0.33], [0.1, -0.33],
    ], device=device, dtype=torch.float32)
    polygon_constraints.append(((om_ind, w0_ind), poly_om_w0))

    return PolytopeBoxUniform(
        low=low, high=high,
        polygon_constraints=polygon_constraints,
        device=device,
    )


# ---------------------------------------------------------------------------
# Log-posterior helper
# ---------------------------------------------------------------------------

def _log_posterior_batch(
    nle_net: torch.nn.Module,
    prior: PolytopeBoxUniform,
    x_obs_tmpl: torch.Tensor,   # (1, 1, data_dim) — broadcast template
    theta_batch: torch.Tensor,  # (n_active, ndim)
) -> torch.Tensor:
    """Return log p(x_obs | theta_i) + log p(theta_i) for each walker i.

    The nflows-based NLE network from likelihood_nn expects:
        log_prob(input=(1, batch, data_dim), condition=(batch, ndim))
    and returns (1, batch); indexing [0] gives (batch,).

    x_obs_tmpl is expanded to (1, n_active, data_dim) without copying data.
    """
    n_active = theta_batch.shape[0]
    x_expanded = x_obs_tmpl.expand(1, n_active, -1)   # (1, n_active, data_dim)
    log_lik    = nle_net.log_prob(x_expanded, condition=theta_batch)[0]   # (n_active,)
    log_prior  = prior.log_prob(theta_batch)                              # (n_active,)
    return log_lik + log_prior


# ---------------------------------------------------------------------------
# GPU ensemble MCMC  (adapted from NLE.get_posterior_gpu in ml_methods.py)
# ---------------------------------------------------------------------------

def sample_posterior_gpu(
    nle_net: torch.nn.Module,
    prior: PolytopeBoxUniform,
    x_obs: torch.Tensor,
    theta_0: np.ndarray | None = None,
    mcmc_params: dict | None = None,
    return_log_prob: bool = False,
    device: str = "cuda:0",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """GPU affine-invariant ensemble MCMC (Goodman & Weare 2010 stretch move).

    All walkers in each complementary half advance in parallel on the GPU,
    avoiding the CPU↔GPU round-trips of the emcee-based method.

    Parameters
    ----------
    nle_net        : Stage-2 NLE network loaded from cloudpickle pkl.
    prior          : PolytopeBoxUniform prior.
    x_obs          : (data_dim,) tensor — compressed C3NN summary of the observation.
    theta_0        : (nwalkers, ndim) initial walker positions, or None → draw from prior.
    mcmc_params    : dict with optional keys:
                       nwalkers  (int)   [1000]
                       nsteps    (int)   [1000]   kept steps after burn-in & thinning
                       discard   (int)   [5000]   burn-in steps
                       thin      (int)   [3]      thinning interval
                       stretch_a (float) [2.0]    stretch-move scale
                       progress  (bool)  [True]   tqdm progress bar
    return_log_prob: if True also return per-sample log-posteriors.
    device         : torch device string.

    Returns
    -------
    samples   : ndarray (n_samples, ndim)
    log_probs : ndarray (n_samples,)   [only when return_log_prob=True]
    """
    p = mcmc_params or {}
    nwalkers  = p.get("nwalkers",  1000)
    nsteps    = p.get("nsteps",    1000)
    discard   = p.get("discard",   5000)
    thin      = p.get("thin",      3)
    stretch_a = p.get("stretch_a", 2.0)
    progress  = p.get("progress",  True)

    if nwalkers % 2 != 0:
        raise ValueError(f"nwalkers must be even; got {nwalkers}.")

    tot_steps = nsteps * thin + discard
    ndim = prior.event_shape[0]

    # Decide whether to keep the full chain in GPU memory or stream only the
    # post-burn-in thinned samples. Uses at most half of available VRAM.
    chain_bytes = tot_steps * nwalkers * ndim * 4
    if "cuda" in str(device):
        free_bytes, _ = torch.cuda.mem_get_info(device)
        store_all = chain_bytes < free_bytes // 2
    else:
        store_all = True  # CPU: no tight memory limit

    T = tot_steps if store_all else nsteps
    # Allocate OUTSIDE inference_mode so these remain regular tensors that
    # can be written to from inside the inference_mode block.
    chain    = torch.empty((T, nwalkers, ndim), device=device)
    lp_chain = torch.empty((T, nwalkers),        device=device) if return_log_prob else None

    step_iter: range | trange = range(tot_steps)
    if progress and trange is not None:
        step_iter = trange(tot_steps, desc="C3NN GPU ensemble MCMC")

    # Pre-build broadcast template: (1, 1, data_dim) — expanded per batch
    x_obs_tmpl = x_obs.to(device).reshape(1, 1, -1)

    half   = nwalkers // 2
    idx_s0 = torch.arange(0,    half,     device=device)
    idx_s1 = torch.arange(half, nwalkers, device=device)

    kept = 0
    with torch.inference_mode():
        # Initialise walker positions
        if theta_0 is not None:
            walkers = torch.as_tensor(theta_0, device=device, dtype=torch.float32)
        else:
            walkers = prior.sample((nwalkers,)).to(device)

        log_probs = _log_posterior_batch(nle_net, prior, x_obs_tmpl, walkers)

        for step in step_iter:
            for active_idx, comp_idx in [(idx_s0, idx_s1), (idx_s1, idx_s0)]:
                n_active = active_idx.shape[0]
                n_comp   = comp_idx.shape[0]

                # Pick a random partner from the complementary half
                j        = torch.randint(n_comp, (n_active,), device=device)
                partners = walkers[comp_idx[j]]

                # Stretch-move scale drawn from g(z) ∝ 1/√z on [1/a, a]
                u = torch.rand(n_active, device=device)
                z = ((stretch_a - 1.0) * u + 1.0) ** 2 / stretch_a

                proposals = partners + z.unsqueeze(1) * (walkers[active_idx] - partners)

                lp_prop = _log_posterior_batch(nle_net, prior, x_obs_tmpl, proposals)

                # Metropolis–Hastings acceptance with the stretch-move Jacobian
                log_accept = (ndim - 1) * torch.log(z) + lp_prop - log_probs[active_idx]
                accept = torch.log(torch.rand(n_active, device=device)) < log_accept

                walkers[active_idx[accept]]    = proposals[accept]
                log_probs[active_idx[accept]]  = lp_prop[accept]

            # Store step
            if store_all:
                chain[step] = walkers
                if return_log_prob:
                    lp_chain[step] = log_probs
            else:
                if step >= discard and (step - discard) % thin == 0 and kept < nsteps:
                    chain[kept] = walkers
                    if return_log_prob:
                        lp_chain[kept] = log_probs
                    kept += 1

    # Outside inference_mode: slice/reshape the regular ``chain`` buffer.
    if store_all:
        samples = chain[discard::thin].reshape(-1, ndim)
        if return_log_prob:
            lp_samples = lp_chain[discard::thin].reshape(-1)
    else:
        samples = chain[:kept].reshape(-1, ndim)
        if return_log_prob:
            lp_samples = lp_chain[:kept].reshape(-1)

    if return_log_prob:
        return samples.cpu().numpy(), lp_samples.cpu().numpy()
    return samples.cpu().numpy()


# ---------------------------------------------------------------------------
# On-the-fly map compression (optional)
# ---------------------------------------------------------------------------

def compress_observation(
    compressor_path: str,
    maps_path: str,
    twopcf_path: str,
    device: str,
) -> torch.Tensor:
    """Compress raw observed maps into a C3NN summary vector.

    Parameters
    ----------
    compressor_path : path to *_c3nn_stage1_compressor.pkl.
    maps_path       : .npy of shape (n_maps, 4, H, W) — 4-channel convergence maps.
    twopcf_path     : .npy of shape (n_2pcf,) — compressed 2PCF coefficients.
    device          : target device.

    Returns
    -------
    x_obs : (out_dim,) tensor on ``device``.
    """
    from C3NN_MDN_compressor import compute_valid_masks  # noqa: F401

    print(f"Loading compressor from {compressor_path} ...")
    with open(compressor_path, "rb") as fh:
        compressor = cpk.load(fh)
    compressor.to(device).eval()

    maps   = torch.tensor(np.load(maps_path), dtype=torch.float32)  # (n_maps, 4, H, W)
    twopcf = torch.tensor(np.load(twopcf_path), dtype=torch.float32)  # (n_2pcf,)

    n_maps, _, H, W = maps.shape
    n_2pcf = twopcf.shape[0]

    # Replicate 2PCF and pad to fill H×W for the 5th channel
    twopcf_rep = twopcf.unsqueeze(0).expand(n_maps, -1)        # (n_maps, n_2pcf)
    zeros_pad  = torch.zeros(n_maps, H * W - n_2pcf)
    padded     = torch.cat([twopcf_rep, zeros_pad], dim=1).reshape(n_maps, 1, H, W)
    maps_5ch   = torch.cat([maps, padded], dim=1)               # (n_maps, 5, H, W)

    # Compute valid-position masks from this observation's footprint
    rep_maps = maps[:, 0]  # (n_maps, H, W) — first channel
    valid_masks = compute_valid_masks(rep_maps, compressor.backbone.filter_size)
    compressor.backbone.set_valid_masks(valid_masks.to(device))

    # Add batch dimension: (1, n_maps, 5, H, W)
    x_batch = maps_5ch.unsqueeze(0).to(device)
    with torch.inference_mode():
        x_obs = compressor.compress(x_batch).squeeze(0)  # (out_dim,)

    return x_obs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the C3NN NLE posterior with GPU ensemble MCMC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Compression mode ---------------------------------------------------
    parser.add_argument(
        "--compression-mode",
        choices=["joint", "separate"],
        default="joint",
        help=(
            "How the observed map is compressed into the NLE summary vector. "
            "'joint': the FC compression layer takes higher-order C3NN features "
            "AND the MOPED 2PCF coefficients together, producing a fixed "
            "(compressed_dim,) summary. "
            "'separate': the FC layer compresses only the higher-order C3NN "
            "features; the MOPED 2PCF coefficients are then concatenated, "
            "producing a (compressed_dim + n_2pcf,) summary. This matches "
            "exactly how train_c3nn_ddp.py and train_nle.py train the models "
            "(output files carry the _moped_separate_ suffix). "
            "Default paths for all model files are set automatically from "
            "this flag; override any path individually with the explicit flags."
        ),
    )

    # --- Model paths --------------------------------------------------------
    parser.add_argument("--nle-net", default=None,
                        help="Path to Stage-2 NLE pkl file. "
                             "Defaults to the mode-appropriate path under _MODEL_DIR.")
    parser.add_argument("--nle-params", default=None,
                        help="Path to training-parameter .npy (used to build the prior). "
                             "Defaults to the mode-appropriate path under _MODEL_DIR.")
    parser.add_argument("--nle-summaries", default=None,
                        help="Path to training-summaries .npy (used with --x-obs-index). "
                             "Defaults to the mode-appropriate path under _MODEL_DIR.")

    # --- Observation --------------------------------------------------------
    obs_group = parser.add_mutually_exclusive_group()
    obs_group.add_argument("--x-obs", default=None,
                           help="Pre-computed C3NN summary .npy, shape (data_dim,) or (1, data_dim).")
    obs_group.add_argument("--x-obs-index", type=int, default=None,
                           help="Row index into --nle-summaries to use as the observation "
                                "(useful for validation / closure tests).")

    # On-the-fly map compression
    parser.add_argument("--compressor", default=None,
                        help="Stage-1 compressor pkl. Required when using --maps. "
                             "Defaults to the mode-appropriate path under _MODEL_DIR "
                             "when --maps is given but --compressor is omitted.")
    parser.add_argument("--maps", default=None,
                        help="Observed maps .npy, shape (n_maps, 4, H, W).")
    parser.add_argument("--twopcf", default=None,
                        help="Compressed 2PCF vector .npy, shape (n_2pcf,).")

    # --- Output -------------------------------------------------------------
    parser.add_argument("--output", default="posterior_samples_c3nn.npy",
                        help="Output path for posterior samples.")
    parser.add_argument("--output-logprob", default=None,
                        help="If set, also save per-sample log-posteriors here.")

    # --- MCMC parameters ----------------------------------------------------
    parser.add_argument("--nwalkers",  type=int,   default=1000)
    parser.add_argument("--nsteps",    type=int,   default=1000,
                        help="Number of kept steps per walker (after burn-in and thinning).")
    parser.add_argument("--discard",   type=int,   default=5000,
                        help="Burn-in steps to discard.")
    parser.add_argument("--thin",      type=int,   default=3,
                        help="Thinning interval.")
    parser.add_argument("--stretch-a", type=float, default=2.0,
                        help="Stretch-move scale parameter a (Goodman & Weare 2010).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress the tqdm progress bar.")

    # --- Misc ---------------------------------------------------------------
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda:0 if available, else cpu).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mode   = args.compression_mode
    defs   = _DEFAULTS[mode]

    print(f"Device           : {device}")
    print(f"Compression mode : {mode}  "
          f"({'higher-order + 2PCF jointly compressed to fixed dim' if mode == 'joint' else 'higher-order compressed, 2PCF appended separately'})")

    # Resolve model-path defaults based on compression mode; explicit CLI
    # flags override the mode defaults.
    nle_net_path       = args.nle_net       or defs["nle_net"]
    nle_params_path    = args.nle_params    or defs["nle_params"]
    nle_summaries_path = args.nle_summaries or defs["nle_summaries"]
    compressor_path    = args.compressor    or defs["compressor"]

    # ---- Load NLE network --------------------------------------------------
    print(f"Loading NLE network  : {nle_net_path}")
    with open(nle_net_path, "rb") as fh:
        nle_net = cpk.load(fh)
    nle_net.to(device).eval()

    # ---- Build prior -------------------------------------------------------
    print(f"Loading NLE params   : {nle_params_path}")
    theta_train = np.load(nle_params_path)          # (N, 5)
    print(f"  theta shape: {theta_train.shape}  params: {PARAM_NAMES}")
    prior = build_prior(theta_train, device)

    # ---- Obtain the observation summary ------------------------------------
    if args.x_obs is not None:
        x_obs = torch.tensor(
            np.load(args.x_obs), dtype=torch.float32
        ).flatten().to(device)
        print(f"Loaded x_obs from    : {args.x_obs}  shape: {tuple(x_obs.shape)}")

    elif args.x_obs_index is not None:
        summaries = np.load(nle_summaries_path)
        idx = args.x_obs_index
        x_obs = torch.tensor(summaries[idx], dtype=torch.float32).to(device)
        print(f"Using training summary index {idx} as observation  shape: {tuple(x_obs.shape)}")
        print(f"  True θ: {dict(zip(PARAM_NAMES, theta_train[idx].tolist()))}")

    elif args.maps is not None:
        if args.twopcf is None:
            raise ValueError("--twopcf is required when using --maps.")
        x_obs = compress_observation(
            compressor_path, args.maps, args.twopcf, device
        )
        print(f"Compressed observation shape: {tuple(x_obs.shape)}")

    else:
        # Fall back to first training summary for a quick sanity check
        summaries = np.load(nle_summaries_path)
        x_obs = torch.tensor(summaries[0], dtype=torch.float32).to(device)
        print("No observation specified — using training summary index 0 as default.")
        print(f"  True θ: {dict(zip(PARAM_NAMES, theta_train[0].tolist()))}")

    # ---- Initial walker positions ------------------------------------------
    nwalkers = args.nwalkers
    print(f"\nInitialising {nwalkers} walkers from the prior ...")
    theta_0 = prior.sample((nwalkers,)).cpu().numpy()   # (nwalkers, ndim)

    # ---- Run MCMC ----------------------------------------------------------
    mcmc_params = {
        "nwalkers":  nwalkers,
        "nsteps":    args.nsteps,
        "discard":   args.discard,
        "thin":      args.thin,
        "stretch_a": args.stretch_a,
        "progress":  not args.no_progress,
    }
    tot_steps = args.nsteps * args.thin + args.discard
    print(f"MCMC params  : {mcmc_params}")
    print(f"Total steps  : {tot_steps}  expected samples: {nwalkers * args.nsteps}")

    return_lp = args.output_logprob is not None
    result = sample_posterior_gpu(
        nle_net=nle_net,
        prior=prior,
        x_obs=x_obs,
        theta_0=theta_0,
        mcmc_params=mcmc_params,
        return_log_prob=return_lp,
        device=device,
    )

    if return_lp:
        samples, log_probs = result
    else:
        samples = result

    # ---- Report and save ---------------------------------------------------
    print(f"\nDone.  Samples shape: {samples.shape}")
    print("Posterior marginals:")
    for i, name in enumerate(PARAM_NAMES):
        v = samples[:, i]
        print(f"  {name:8s}  mean={v.mean():.4f}  std={v.std():.4f}"
              f"  [{v.min():.4f}, {v.max():.4f}]")

    np.save(args.output, samples)
    print(f"\nSamples saved to {args.output}")

    if return_lp:
        np.save(args.output_logprob, log_probs)
        print(f"Log-posteriors saved to {args.output_logprob}")


if __name__ == "__main__":
    main()
