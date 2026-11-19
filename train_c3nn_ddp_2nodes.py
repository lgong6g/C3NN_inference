"""Stage 1: Train C3NN + MDN-VMIM compressor with DDP.

After training completes, compressed summaries are extracted from the NLE
training maps using the frozen compressor (extraction is distributed across
all GPUs) and saved to disk for use by train_nle.py.

Outputs written by rank 0:
  {model_dir}/{tag}_compressor.pkl          -- cloudpickle of final compressor
  {model_dir}/{tag}_nle_summaries.npy       -- float32 array [N_nle, out_dim]
  {model_dir}/{tag}_nle_params.npy          -- float32 array [N_nle, n_params]
  {checkpoint_dir}/c3nn_epoch_{NNNN}.pt     -- periodic training checkpoints

Launch (single node, N GPUs):
    torchrun --nproc_per_node=N train_c3nn_ddp.py [--resume] [--test]

Multi-node example (2 nodes, 4 GPUs each):
    # node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=<node0_ip> --master_port=12355 train_c3nn_ddp.py
    # node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
             --master_addr=<node0_ip> --master_port=12355 train_c3nn_ddp.py

To resume an interrupted run, add --resume.  The script picks up from the
latest checkpoint in the checkpoint directory automatically.
"""

import sys
sys.path.append('./src')

import gc
import os
import math
import glob
import argparse
import time
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
import cloudpickle as cpk

from hyperparameters import hparams
from load_data import load_partitioned_mmap
from C3NN_MDN_compressor import (
    C3NN_hybrid_highorder,
    C3NN_MDN_Compressor,
    mdn_nll_loss,
    compute_valid_masks,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HybridMmapDataset(Dataset):
    """Streams memory-mapped maps from disk and appends 2PCF on the fly."""

    def __init__(self, maps_mmap, params, compressed_2pt):
        self.maps_mmap = maps_mmap
        self.params = torch.tensor(params, dtype=torch.float32)
        self.compressed_2pt = torch.tensor(compressed_2pt, dtype=torch.float32)
        self.num_cosmo, self.num_maps, self.C, self.H, self.W = maps_mmap.shape

    def __len__(self):
        return self.num_cosmo

    def __getitem__(self, idx):
        map_data = torch.tensor(self.maps_mmap[idx], dtype=torch.float32)
        c_2pt = self.compressed_2pt[idx]
        c_2pt_repeated = c_2pt.repeat(self.num_maps, 1)
        zeros_pad = torch.zeros(
            (self.num_maps, self.H * self.W - c_2pt.shape[0]), dtype=torch.float32
        )
        padded = torch.cat((c_2pt_repeated, zeros_pad), dim=1)
        padded_square = padded.reshape((self.num_maps, 1, self.H, self.W))
        return torch.cat((map_data, padded_square), dim=1), self.params[idx]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path of the highest-numbered checkpoint, or None."""
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "c3nn_epoch_*.pt")))
    return files[-1] if files else None


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    compressor_ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_loss: float,
    best_state: dict | None,
    patience_ctr: int,
    rank: int,
) -> None:
    """Persist a training checkpoint (rank 0 only)."""
    if rank != 0:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"c3nn_epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": compressor_ddp.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "patience_ctr": patience_ctr,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# Stage 1 training
# ---------------------------------------------------------------------------

def train_c3nn_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    hparams,
    maps_mmap,
    training_parameters: np.ndarray,
    training_2pt: np.ndarray,
    checkpoint_dir: str,
    n_components: int = 8,
    max_epochs: int = 300,
    early_stop_patience: int = 10,
    checkpoint_every: int = 20,
    resume: bool = False,
    verbose: bool = True,
    test: bool = False,
    compression_mode: str = "separate",
):
    """Train C3NN + MDN-VMIM compressor with DDP.

    compression_mode controls how the MOPED 2PCF coefficients are incorporated:
      'separate' — FC layer compresses only the higher-order C3NN features;
                   2PCF is appended afterwards → summary dim = out_dim + n_2pcf.
      'joint'    — FC layer receives (higher-order + 2PCF) together and compresses
                   them into a single fixed-length vector → summary dim = out_dim.

    Returns the underlying C3NN_MDN_Compressor (weights frozen, on CPU).
    """
    device = torch.device(f"cuda:{local_rank}")
    n_params      = training_parameters.shape[1]
    # N_2PCF        = 17 # 17 is the ndim of the full MOPED coefficients
    N_2PCF = 5
    highorder_dim = (hparams.order - 2) * hparams.num_filts

    if compression_mode == "joint":
        # FC input = higher-order features + 2PCF; output = compressed_dim only
        raw_dim    = highorder_dim + N_2PCF
        n_2pcf_arg = 0
    else:  # separate
        # FC input = higher-order features only; 2PCF appended after FC
        raw_dim    = highorder_dim
        n_2pcf_arg = N_2PCF

    # ---- Build model -------------------------------------------------------
    backbone = C3NN_hybrid_highorder(hparams, n_2pcf=N_2PCF).to(device)
    rep_maps = torch.from_numpy(np.array(maps_mmap[0, :, 0])).float()
    backbone.set_valid_masks(compute_valid_masks(rep_maps, hparams.filter_size).to(device))

    compressor = C3NN_MDN_Compressor(
        backbone=backbone,
        raw_dim=raw_dim,
        compressed_dim=hparams.out_dim,
        n_params=n_params,
        n_components=n_components,
        n_2pcf=n_2pcf_arg,
    ).to(device)
    compressor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(compressor)
    compressor_ddp = DDP(compressor, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.Adam(compressor_ddp.parameters(), lr=hparams.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    # ---- Data loaders ------------------------------------------------------
    N     = maps_mmap.shape[0]
    n_val = max(1, int(0.2 * N))

    train_dataset = HybridMmapDataset(
        maps_mmap[: N - n_val], training_parameters[: N - n_val], training_2pt[: N - n_val]
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_dataset, batch_size=hparams.c3nn_batch_size,
        sampler=train_sampler, pin_memory=True, num_workers=16, persistent_workers=True,
    )

    val_dataset = HybridMmapDataset(
        maps_mmap[N - n_val :], training_parameters[N - n_val :], training_2pt[N - n_val :]
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader  = DataLoader(
        val_dataset, batch_size=hparams.c3nn_batch_size,
        sampler=val_sampler, pin_memory=True, num_workers=8, persistent_workers=True,
    )

    # ---- Resume from checkpoint (if requested) -----------------------------
    start_epoch   = 0
    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    if resume:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            # All ranks load from the shared filesystem simultaneously.
            ckpt = torch.load(ckpt_path, map_location=device)
            compressor_ddp.module.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            best_state    = ckpt.get("best_state")
            patience_ctr  = ckpt["patience_ctr"]
            if verbose and rank == 0:
                print(f"Resumed from checkpoint: {ckpt_path}  (epoch {start_epoch})")
        else:
            if verbose and rank == 0:
                print("--resume requested but no checkpoint found; starting fresh.")

    # ---- Training loop -----------------------------------------------------
    # FP32 + TF32 (no autocast/GradScaler): K = 4*37*37 = 5476 is divisible
    # by 4 but NOT by 8, so FP16 tensor cores are misaligned and ~2.2x slower.
    # TF32 (K%4==0) is optimal here; GradScaler is not needed for FP32.
    n_epochs = 3 if test else max_epochs

    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        compressor_ddp.train()

        epoch_loss = 0.0
        n_batches  = 0
        for xb, tb in train_loader:
            xb, tb = xb.to(device, non_blocking=True), tb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pi_logits, mu, log_sigma = compressor_ddp(xb)
            loss = mdn_nll_loss(pi_logits, mu, log_sigma, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(compressor_ddp.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        # Validation (all ranks, then all_reduce)
        compressor_ddp.eval()
        val_loss_sum  = torch.tensor(0.0, device=device)
        val_n_batches = torch.tensor(0, dtype=torch.long, device=device)
        with torch.no_grad():
            for xb_val, tb_val in val_loader:
                xb_val, tb_val = xb_val.to(device), tb_val.to(device)
                pi_v, mu_v, ls_v = compressor_ddp.module(xb_val)
                val_loss_sum  += mdn_nll_loss(pi_v, mu_v, ls_v, tb_val)
                val_n_batches += 1
        dist.all_reduce(val_loss_sum,  op=dist.ReduceOp.SUM)
        dist.all_reduce(val_n_batches, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_sum / val_n_batches.clamp(min=1)).item()
        scheduler.step(val_loss)

        avg_train = epoch_loss / max(1, n_batches)
        if verbose and rank == 0 and (epoch % 10 == 0 or epoch < 5):
            print(
                f"Stage 1 | Epoch {epoch:4d} | "
                f"train NLL {avg_train:.4f} | val NLL {val_loss:.4f}"
            )

        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                best_state = {k: v.cpu().clone() for k, v in compressor_ddp.module.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience and not test:
                if verbose and rank == 0:
                    print(f"Stage 1: early stopping at epoch {epoch} (best val NLL {best_val_loss:.4f})")
                # Save a final checkpoint before breaking
                save_checkpoint(
                    checkpoint_dir, epoch, compressor_ddp, optimizer, scheduler,
                    best_val_loss, best_state, patience_ctr, rank,
                )
                break

        # Periodic checkpoint every `checkpoint_every` epochs
        if rank == 0 and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir, epoch, compressor_ddp, optimizer, scheduler,
                best_val_loss, best_state, patience_ctr, rank,
            )

    # ---- Broadcast best weights to all ranks ------------------------------
    state_dict_list = [None]
    if rank == 0 and best_state is not None:
        state_dict_list[0] = {k: v.cpu() for k, v in best_state.items()}
    dist.broadcast_object_list(state_dict_list, src=0)
    compressor_ddp.module.load_state_dict(
        {k: v.to(device) for k, v in state_dict_list[0].items()}
    )

    if verbose and rank == 0:
        print(f"Stage 1 complete. Best val NLL: {best_val_loss:.4f}")

    # Freeze and return underlying module (on CPU for pickling)
    module = compressor_ddp.module
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module.cpu(), best_val_loss


# ---------------------------------------------------------------------------
# Distributed summary extraction for NLE training data
# ---------------------------------------------------------------------------

def extract_summaries_distributed(
    compressor,
    maps_mmap,
    params: np.ndarray,
    compressed_2pt: np.ndarray,
    rank: int,
    local_rank: int,
    world_size: int,
    out_dim: int,
    batch_size: int = 32,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract compressed summaries from all NLE maps (distributed).

    Each rank processes a contiguous slice of the dataset; dist.all_gather
    reassembles the full [N, out_dim] summary matrix on every rank.

    Returns (summaries, params) as float32 numpy arrays on rank 0.
    Other ranks return (None, None).

    batch_size=32: with n_maps=3 the flat batch is [96, 4, 512, 512].
    NonlinearConv keeps 13 simultaneous tensors of that shape in the two
    accumulation lists (conv_powers + corrs + Cn).  Under autocast (fp16)
    peak usage is 13 × 1.875 GiB = 24.4 GiB — safely within the 40 GiB
    A100 budget.  batch_size=64 (fp32, no autocast) would require 97.5 GiB
    and crashes on the 5th conv_powers entry.
    """
    device = torch.device(f"cuda:{local_rank}")
    N = maps_mmap.shape[0]

    per_rank    = math.ceil(N / world_size)
    local_start = rank * per_rank
    local_end   = min(local_start + per_rank, N)

    if verbose and rank == 0:
        print(f"Extracting NLE summaries (distributed, {N} cosmologies) ...")

    full_dataset  = HybridMmapDataset(maps_mmap, params, compressed_2pt)
    local_dataset = Subset(full_dataset, range(local_start, local_end))
    local_loader  = DataLoader(
        local_dataset, batch_size=batch_size,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    compressor = compressor.to(device)
    compressor.eval()

    local_sums = torch.zeros(per_rank, out_dim, device=device)
    
    with torch.no_grad():
        idx = 0
        for xb, _ in local_loader:
            xb = xb.to(device, non_blocking=True)
            batch_n = xb.shape[0]
            with autocast("cuda"):
                sums = compressor.compress(xb)
            local_sums[idx : idx + batch_n] = sums.float()
            idx += batch_n

    """
    with torch.no_grad():
      idx = 0
      for xb, _ in local_loader:
          xb = xb.to(device, non_blocking=True)
          batch_n = xb.shape[0]
          sums = compressor.compress(xb)   # fp32 — no underflow possible
          local_sums[idx : idx + batch_n] = sums
          idx += batch_n
    """
    # Flush all pending CUDA writes to local_sums before NCCL reads it.
    torch.cuda.synchronize()

    full_tensor = torch.zeros(world_size * per_rank, out_dim, device=device)
    dist.all_gather_into_tensor(full_tensor, local_sums)

    # Flush NCCL's internal stream writes to full_tensor before the default
    # stream reads it.  dist.all_gather_into_tensor is CPU-blocking (all ranks
    # have entered the collective when it returns) but the actual GPU writes to
    # full_tensor happen on NCCL's own CUDA stream, which is separate from the
    # default stream.  Without this sync the (full_tensor == 0) comparison
    # kernel on the default stream races against NCCL's writes and reads
    # whichever rows NCCL hasn't finished yet — still holding the torch.zeros
    # initialization — producing hundreds of spurious all-zero rows.
    torch.cuda.synchronize()

    if rank == 0:
        summaries_tensor = full_tensor[:N]
        zero_rows = (summaries_tensor == 0).all(dim=1).sum().item()
        if zero_rows > 10:
            print(
                f"WARNING: ALLGATHER produced {zero_rows} all-zero rows — "
                f"likely a partial NCCL failure. "
                f"Summary extraction aborted; compressor will still be saved."
            )
            return None, None
        summaries = summaries_tensor.cpu().numpy().astype(np.float32)
        if verbose:
            print(f"Summaries extracted. Shape: {summaries.shape}")
        return summaries, params.astype(np.float32)
    return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Train C3NN + MDN-VMIM compressor (DDP). "
                    "Launch with torchrun."
    )
    parser.add_argument("-t", "--test", action="store_true",
                        help="Smoke-test mode (3 epochs, small data subset)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    parser.add_argument("--mdn-components", type=int, default=8,
                        help="MDN Gaussian components (default: 8)")
    parser.add_argument("--master-port", type=str, default="12355",
                        help="MASTER_PORT for DDP (default: 12355)")
    parser.add_argument(
        "--compression-mode",
        choices=["joint", "separate"],
        default="separate",
        help=(
            "How the MOPED 2PCF coefficients are combined with the higher-order "
            "C3NN features. "
            "'separate' (default): FC layer compresses only the higher-order "
            "features; 2PCF is concatenated afterwards → summary dim = "
            "out_dim + n_2pcf.  Output files carry the _moped_separate_ suffix. "
            "'joint': FC layer compresses higher-order features AND 2PCF together "
            "→ summary dim = out_dim (fixed).  Output files use plain naming."
        ),
    )
    args = parser.parse_args()

    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", args.master_port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=2),
    )
    torch.cuda.set_device(local_rank)

    seed = 1750
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled             = True
    torch.backends.cudnn.benchmark           = True   # profile fastest algorithm on first run
    torch.backends.cudnn.allow_tf32          = True   # TF32 for conv (K=5476 divisible by 4)
    torch.backends.cuda.matmul.allow_tf32    = True   # TF32 for linear layers (off by default here)
    # deterministic=True was previously set but contradicts benchmark=True and disables
    # cuDNN algorithm selection. Removed: non-determinism from float op ordering is acceptable.

    compression_mode = args.compression_mode

    model_dir = (
        "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models/codetesting"
        if args.test
        else "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models"
    )
    # Separate checkpoint directories per mode so joint and separate runs
    # never share or overwrite each other's checkpoints.
    checkpoint_dir = os.path.join(model_dir, "checkpoints", f"c3nn_{compression_mode}")

    time_ini = time.time()

    if rank == 0:
        print("Loading mmaps ...")
    c3nn_indices = np.load("/pscratch/sd/l/lgong/C3NN_inference/data/c3nn_training_data_param_indices.npy")
    nle_indices  = np.load("/pscratch/sd/l/lgong/C3NN_inference/data/nle_training_data_param_indices.npy")

    c3nn_maps, c3nn_params, c3nn_2pt = load_partitioned_mmap(
        "/pscratch/sd/l/lgong/C3NN_inference/data/c3nn_training_maps_mmap_smoothed.npy", c3nn_indices
    )
    nle_maps, nle_params, nle_2pt = load_partitioned_mmap(
        "/pscratch/sd/l/lgong/C3NN_inference/data/nle_training_maps_mmap_smoothed.npy", nle_indices
    )

    if args.test:
        c3nn_maps   = c3nn_maps[:3200]
        c3nn_params = c3nn_params[:3200]
        c3nn_2pt    = c3nn_2pt[:3200]
        nle_maps    = nle_maps[:3200]
        nle_params  = nle_params[:3200]
        nle_2pt     = nle_2pt[:3200]

    if args.test and rank == 0:
        print("Running in test mode (3 epochs, small subset)")

    # ---- Stage 1: train compressor ----------------------------------------
    if rank == 0:
        print("=" * 60)
        print(f"STAGE 1: C3NN + MDN-VMIM  [world_size={world_size}]")
        print(f"  Compression mode : {compression_mode}")
        print("=" * 60)

    compressor, best_val_loss = train_c3nn_ddp(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        hparams=hparams,
        maps_mmap=c3nn_maps,
        training_parameters=c3nn_params,
        training_2pt=c3nn_2pt,
        checkpoint_dir=checkpoint_dir,
        n_components=args.mdn_components,
        checkpoint_every=20,
        resume=args.resume,
        test=args.test,
        compression_mode=compression_mode,
    )

    # Release DDP wrapper, optimizer, and scaler memory before extraction.
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Extract NLE summaries (distributed) ------------------------------
    if rank == 0:
        print("=" * 60)
        print("Extracting NLE training summaries ...")
        print("=" * 60)

    N_2PCF = 5
    if compression_mode == "separate":
        out_dim     = hparams.out_dim + N_2PCF   # [N, compressed_dim + n_2pcf]
        #mode_suffix = "_moped_separate" # This suffix is for the 17 moped coeffcients case
        mode_suffix = "_moped_separate_5coeffs"
    else:  # joint
        out_dim     = hparams.out_dim             # [N, compressed_dim]
        #mode_suffix = "" # This suffix is for the 17 moped coefficients case
        mode_suffix = "_5coeffs"
    summaries, saved_params = extract_summaries_distributed(
        compressor=compressor,
        maps_mmap=nle_maps,
        params=nle_params,
        compressed_2pt=nle_2pt,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        out_dim=out_dim,
    )

    # ---- Save (rank 0 only) -----------------------------------------------
    if rank == 0 and hparams.save:
        os.makedirs(model_dir, exist_ok=True)
        tag = hparams.name_model() + "_c3nn_stage1"

        compressor_path = os.path.join(model_dir, f"{tag}{mode_suffix}_compressor.pkl") # This is for training on 2 gpu nodes with 40gb A100
        with open(compressor_path, "wb") as fh:
            cpk.dump(compressor, fh)
        print(f"Compressor saved  → {compressor_path}")

        if summaries is not None:
            summaries_path = os.path.join(model_dir, f"{tag}_nle{mode_suffix}_summaries.npy") # training with 40gb A100
            params_path    = os.path.join(model_dir, f"{tag}_nle{mode_suffix}_params.npy") # training with 40gb A100
            np.save(summaries_path, summaries)
            np.save(params_path, saved_params)
            print(f"NLE summaries     → {summaries_path}")
            print(f"NLE params        → {params_path}")
            print(f"Pass these paths to train_nle.py via --summaries and --params.")
        else:
            print("NLE summaries skipped (extraction failed — rerun extraction separately).")
    elif rank == 0:
        print("hparams.save=False — skipping file output.")

    dist.destroy_process_group()

    if rank == 0:
        elapsed = datetime.timedelta(seconds=time.time() - time_ini)
        print(f"Stage 1 finished. Elapsed: {elapsed}")
