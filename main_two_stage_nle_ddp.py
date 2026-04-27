"""Two-stage NLE training with DistributedDataParallel (DDP) across multiple GPUs.

Launch (single node, N GPUs):
    torchrun --nproc_per_node=N main_two_stage_nle_ddp.py [--test] [--mdn-components K]

Multi-node example (2 nodes, 4 GPUs each):
    # On node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
             --master_addr=<node0_ip> --master_port=12355 main_two_stage_nle_ddp.py
    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\
             --master_addr=<node0_ip> --master_port=12355 main_two_stage_nle_ddp.py

Differences from main_two_stage_nle.py (single-GPU)
-----------------------------------------------------
Stage 1 — C3NN + MDN-VMIM:
  * DistributedSampler shards the training set across GPUs.
  * Effective batch size = flow_training_batch_size * world_size.
  * DDP averages gradients after every backward pass.
  * Validation loss is computed on rank 0 and broadcast so every rank makes
    the same early-stopping decision.

Stage 2 — NLE:
  * Summary extraction: each GPU processes a disjoint slice of the full
    training set; dist.all_gather reassembles the complete matrix on every rank.
  * NLE training: a custom DDP loop using the MAF network built by sbi's
    likelihood_nn builder (bypasses sbi's SNLE.train() which has no DDP support).
  * The nflows/pyknos Flow.forward(x, context=theta) returns log p(x|theta),
    so DDP gradient hooks work identically to any other nn.Module.

We do not need to parallelize the NLE training. We can make the compression of the NLE training data outside of the 2nd stage NLE training

We also need to perform the hyper-tuning in a two-stage approach
Saving: only rank 0 writes model files.
"""

import sys
sys.path.append('./src')

import os
import math
import argparse
import time
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import cloudpickle as cpk
from torch import from_numpy

from sbi.neural_nets import likelihood_nn
from sbi import utils

from hyperparameters import hparams
from load_data import load_partitioned_mmap
from C3NN_MDN_compressor import (
    C3NN_hybrid_highorder,
    C3NN_MDN_Compressor,
    mdn_nll_loss,
    compute_valid_masks,
)

class HybridMmapDataset(Dataset):
    """Streams memory-mapped maps from disk and appends 2PCF on the fly."""
    def __init__(self, maps_mmap, params, compressed_2pt):
        self.maps_mmap = maps_mmap  # Keep as np.memmap
        self.params = torch.tensor(params, dtype=torch.float32)
        self.compressed_2pt = torch.tensor(compressed_2pt, dtype=torch.float32)
        self.num_cosmo, self.num_maps, self.C, self.H, self.W = maps_mmap.shape

    def __len__(self):
        return self.num_cosmo

    def __getitem__(self, idx):
        # Read ONLY this cosmology's maps from disk
        map_data = torch.tensor(self.maps_mmap[idx], dtype=torch.float32)
        
        # Format 2PCF and create padding
        c_2pt = self.compressed_2pt[idx]
        c_2pt_len = c_2pt.shape[0]
        c_2pt_repeated = c_2pt.repeat(self.num_maps, 1)
        zeros_pad = torch.zeros((self.num_maps, self.H * self.W - c_2pt_len), dtype=torch.float32)
        
        # Concatenate and reshape 5th channel
        padded = torch.cat((c_2pt_repeated, zeros_pad), dim=1)
        padded_square = padded.reshape((self.num_maps, 1, self.H, self.W))
        
        # Append as the 5th channel
        maps_and_2pt = torch.cat((map_data, padded_square), dim=1)
        
        return maps_and_2pt, self.params[idx]


# =============================================================================
# DDP helpers
# =============================================================================

def setup_ddp(rank: int, local_rank: int, world_size: int, master_port: str = "12355") -> None:
    """Initialise the default NCCL process group."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank) # <-- FIX


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def broadcast_scalar(value: float, rank: int, device: torch.device) -> float:
    """Broadcast a Python float from rank 0 to all ranks."""
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(t, src=0)
    return t.item()

class DDPFlowWrapper(nn.Module):
    """Wraps an sbi flow so PyTorch DDP can hook into its .forward() method."""
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self, x, condition):
        return self.flow.log_prob(input=x.unsqueeze(0), condition=condition)[0]


# =============================================================================
# Stage 1: DDP training of C3NN + MDN-VMIM compressor
# =============================================================================

def train_stage1_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    hparams,
    training_parameters: torch.Tensor,
    maps_mmap: torch.Tensor,
    training_2pt: torch.Tensor,
    n_components: int = 8,
    max_epochs: int = 300,
    early_stop_patience: int = 50,
    verbose: bool = True,
    test: bool = False,
) -> tuple:
    """Train C3NN + MDN-VMIM compressor with DDP (Stage 1).

    Args:
        rank:                 This process's global rank.
        world_size:           Total number of processes.
        hparams:              Hyperparameter container.
        training_parameters:  CPU tensor [N, n_params].
        training_data:        CPU tensor [N, n_maps, 5, H, W].
        n_components:         MDN mixture components.
        max_epochs:           Maximum training epochs.
        early_stop_patience:  Early stopping patience.
        verbose:              Print progress (rank 0 only).
        test:                 Smoke-test (3 epochs).

    Returns:
        (compressor_module, best_val_loss)
        compressor_module is the underlying C3NN_MDN_Compressor (not DDP-wrapped),
        with all weights frozen and ready for Stage 2 extraction.
    """
    device = torch.device(f"cuda:{local_rank}")
    n_params = training_parameters.shape[1]
    N_2PCF   = 17 # this is the dimension of the MOPED coefficients of 2PCFs
    raw_dim  = (hparams.order - 2) * hparams.num_filts + N_2PCF

    # ---- Build model on this rank ----------------------------------------
    backbone   = C3NN_hybrid_highorder(hparams, n_2pcf=N_2PCF).to(device)

    # Compute per-cutout valid-position masks once from the first cosmology.
    # maps_mmap[0, :, 0] -> [n_maps, H, W]: first convergence channel of each cutout.
    # The footprint of each cutout is fixed across all cosmologies.
    rep_maps = torch.from_numpy(np.array(maps_mmap[0, :, 0])).float()
    backbone.set_valid_masks(compute_valid_masks(rep_maps, hparams.filter_size).to(device))

    compressor = C3NN_MDN_Compressor(
        backbone=backbone,
        raw_dim=raw_dim,
        compressed_dim=hparams.out_dim,
        n_params=n_params,
        n_components=n_components,
    ).to(device)

    # force BatchNorm layers to synchronize their statistics across the InfiniBand network.
    compressor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(compressor)
    # DDP broadcasts rank-0 parameters to all other ranks on construction.
    # find_unused_parameters=True is safer with vmap-based backbones.
    compressor_ddp = DDP(compressor, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.Adam(compressor_ddp.parameters(), lr=hparams.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, verbose=(verbose and rank == 0)
    )

    # ---- Data split (80 / 20 train / val) --------------------------------
    N     = maps_mmap.shape[0]
    n_val = max(1, int(0.2 * N))
    
    # Train Dataset
    train_dataset = HybridMmapDataset(
        maps_mmap[:N - n_val], training_parameters[:N - n_val], training_2pt[:N - n_val]
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_dataset, batch_size=hparams.c3nn_batch_size,
        sampler=train_sampler, pin_memory=True, num_workers=16, persistent_workers=True
    )

    # Validation is distributed across all ranks to avoid a single-rank bottleneck
    # that would cause other ranks to time out waiting at the broadcast barrier.
    val_dataset = HybridMmapDataset(
        maps_mmap[N - n_val:], training_parameters[N - n_val:], training_2pt[N - n_val:]
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.c3nn_batch_size,
        sampler=val_sampler, pin_memory=True, num_workers=8,
    )

    n_epochs     = 3 if test else max_epochs
    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0
    scaler        = GradScaler('cuda')

    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)   # ensures different shuffles per epoch
        compressor_ddp.train()

        epoch_loss = 0.0
        n_batches  = 0
        for xb, tb in train_loader:
            xb, tb = xb.to(device, non_blocking=True), tb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast('cuda'):
                pi_logits, mu, log_sigma = compressor_ddp(xb)
                loss = mdn_nll_loss(pi_logits, mu, log_sigma, tb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(compressor_ddp.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        # ---- Validation: all ranks compute on their shard, all_reduce -----
        compressor_ddp.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_n_batches = torch.tensor(0, dtype=torch.long, device=device)
        with torch.no_grad(), autocast('cuda'):
            for xb_val, tb_val in val_loader:
                xb_val, tb_val = xb_val.to(device), tb_val.to(device)
                pi_v, mu_v, ls_v = compressor_ddp.module(xb_val)
                val_loss_sum += mdn_nll_loss(pi_v, mu_v, ls_v, tb_val)
                val_n_batches += 1
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_n_batches, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_sum / val_n_batches.clamp(min=1)).item()
        scheduler.step(val_loss)

        avg_train = epoch_loss / max(1, n_batches)
        if verbose and rank == 0 and (epoch % 10 == 0 or epoch < 5):
            print(
                f"Stage 1 | Epoch {epoch:4d} | "
                f"train NLL {avg_train:.4f} | val NLL {val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if rank == 0:
                best_state = {
                    k: v.cpu().clone()
                    for k, v in compressor_ddp.module.state_dict().items()
                }
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience and not test:
                if verbose and rank == 0:
                    print(
                        f"Stage 1: early stopping at epoch {epoch} "
                        f"(best val NLL {best_val_loss:.4f})"
                    )
                break

    # ---- Broadcast entire state_dict (including BatchNorm buffers) --------
    state_dict_list = [None]
    if rank == 0 and best_state is not None:
        state_dict_list[0] = {k: v.cpu() for k, v in best_state.items()}
    
    dist.broadcast_object_list(state_dict_list, src=0)
    compressor_ddp.module.load_state_dict({k: v.to(device) for k, v in state_dict_list[0].items()})

    if verbose and rank == 0:
        print(f"Stage 1 complete. Best val NLL: {best_val_loss:.4f}")

    # Freeze all weights; return the underlying module (not DDP wrapper)
    module = compressor_ddp.module
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()

    return module, best_val_loss


# =============================================================================
# Stage 2: Distributed extraction + DDP NLE training
# =============================================================================

def train_stage2_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    hparams,
    compressor,           # frozen C3NN_MDN_Compressor module
    training_parameters: torch.Tensor,
    maps_mmap: torch.Tensor,
    training_2pt: torch.Tensor,
    verbose: bool = True,
    test: bool = False,
) -> tuple:
    """Train NLE on frozen summaries with DDP (Stage 2).

    Extraction is distributed: each rank processes a disjoint slice of the
    training set and dist.all_gather reassembles the full summary matrix on
    every rank.  The MAF likelihood network is then trained with a custom DDP
    loop using sbi's likelihood_nn builder.

    Args:
        rank, world_size:     Process group info.
        hparams:              Hyperparameter container.
        compressor:           Frozen C3NN_MDN_Compressor (underlying module).
        training_parameters:  CPU tensor [N, n_params].
        training_data:        CPU tensor [N, n_maps, 5, H, W].
        verbose:              Print progress (rank 0 only).
        test:                 Smoke-test (3 epochs).

    Returns:
        (best_val_log_prob, nle_module)
        nle_module is the underlying MAF network (not DDP-wrapped).
    """
    device  = torch.device(f"cuda:{local_rank}")
    N       = maps_mmap.shape[0]
    out_dim = hparams.out_dim

    # ---- Distributed summary extraction ----------------------------------
    # Each rank processes indices [rank, rank+world_size, rank+2*world_size, ...]
    # Padding to equal-length chunks for all_gather.
    per_rank    = math.ceil(N / world_size)
    local_start = rank * per_rank
    local_end   = min(local_start + per_rank, N)

    if verbose and rank == 0:
        print("Extracting Stage-1 compressed summaries (distributed) ...")

    # Create a dataset for just THIS rank's slice
    full_dataset = HybridMmapDataset(maps_mmap, training_parameters, training_2pt)
    local_dataset = Subset(full_dataset, range(local_start, local_end))
    local_loader = DataLoader(local_dataset, batch_size=64, num_workers=4)
    
    compressor.eval()
    local_sums = torch.zeros(per_rank, out_dim, device=device)
    
    with torch.no_grad():
        idx = 0
        for xb, _ in local_loader:
            xb = xb.to(device, non_blocking=True)
            batch_size = xb.shape[0]
            local_sums[idx : idx + batch_size] = compressor.compress(xb)
            idx += batch_size

    # all_gather: every rank receives the full [world_size, per_rank, out_dim] tensor
    gathered = [torch.zeros_like(local_sums) for _ in range(world_size)]
    dist.all_gather(gathered, local_sums)

    # Reassemble in original contiguous order and trim trailing padding
    summaries = torch.cat(gathered, dim=0)[:N].cpu()   # [N, out_dim]
    if verbose and rank == 0:
        print(f"Summaries extracted. Shape: {summaries.shape}")

    # ---- NLE training: rank 0 only ---------------------------------------
    # The NSF trains on [N, out_dim] summaries — tiny tensors whose forward
    # pass is far cheaper than the NCCL all-reduce that DDP would add every
    # backward step.  Ranks 1-3 skip the loop and idle at the barrier below.
    best_val_lp = float("-inf")
    nle_net     = None

    if rank == 0:
        training_parameters_t = torch.tensor(training_parameters, dtype=torch.float32)

        nle_builder = likelihood_nn(
            model="nsf",
            hidden_features=hparams.flow_hidden_features,
            num_transforms=hparams.flow_num_transforms,
            num_blocks=hparams.num_blocks,
            num_bins=hparams.num_bins,
            z_score_x="independent",
            z_score_theta="independent",
        )

        nle_net = nle_builder(
            batch_theta=training_parameters_t,
            batch_x=summaries,
        ).to(device)

        optimizer = optim.Adam(nle_net.parameters(), lr=hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=20, verbose=verbose
        )

        # ---- Data split (80 / 20 train / val) ----------------------------
        n_val_nle = max(1, int(0.2 * N))
        x_nle_tr  = summaries[:N - n_val_nle].to(device)
        t_nle_tr  = training_parameters_t[:N - n_val_nle].to(device)
        x_nle_val = summaries[N - n_val_nle:].to(device)
        t_nle_val = training_parameters_t[N - n_val_nle:].to(device)

        nle_dataset = TensorDataset(t_nle_tr, x_nle_tr)
        nle_loader  = DataLoader(
            nle_dataset,
            batch_size=hparams.flow_training_batch_size,
            shuffle=True,
            num_workers=0,
        )

        n_epochs     = 3 if test else 300
        best_state   = None
        patience_ctr = 0
        patience_nle = 30

        for epoch in range(n_epochs):
            nle_net.train()
            epoch_loss = 0.0
            n_batches  = 0
            for theta_b, x_b in nle_loader:
                optimizer.zero_grad()
                loss = nle_net.loss(x_b, condition=theta_b).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(nle_net.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            nle_net.eval()
            with torch.no_grad():
                val_lp = nle_net.log_prob(
                    x_nle_val.unsqueeze(0), condition=t_nle_val
                )[0].mean().item()

            scheduler.step(val_lp)

            avg_train = epoch_loss / max(1, n_batches)
            if verbose and (epoch % 10 == 0 or epoch < 5):
                print(
                    f"Stage 2 | Epoch {epoch:4d} | "
                    f"train log-prob {-avg_train:.4f} | val log-prob {val_lp:.4f}"
                )

            if val_lp > best_val_lp:
                best_val_lp  = val_lp
                best_state   = {k: v.cpu().clone() for k, v in nle_net.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience_nle and not test:
                    if verbose:
                        print(
                            f"Stage 2: early stopping at epoch {epoch} "
                            f"(best val log-prob {best_val_lp:.4f})"
                        )
                    break

        if best_state is not None:
            nle_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        if verbose:
            print(f"Stage 2 complete. Best val log-prob: {best_val_lp:.4f}")

    # ---- Synchronise: ranks 1-3 idle here while rank 0 trains -----------
    dist.barrier()
    best_val_lp = broadcast_scalar(best_val_lp, rank, device)

    return best_val_lp, nle_net


# =============================================================================
# Top-level DDP entry point
# =============================================================================

def main_ddp(
    rank: int,
    local_rank: int,
    world_size: int,
    hparams,
    # ---- Stage 1 (C3NN) Data ----
    c3nn_maps_mmap: torch.Tensor,
    c3nn_parameters: torch.Tensor,           
    c3nn_2pt: torch.Tensor,                  
    # ---- Stage 2 (NLE) Data -----
    nle_maps_mmap: torch.Tensor,             
    nle_parameters: torch.Tensor,
    nle_2pt: torch.Tensor,
    verbose: bool = True,
    test: bool = False,
    n_mdn_components: int = 8,
) -> tuple:
    """Full two-stage NLE pipeline with DDP.

    Args:
        rank, world_size:     Process group info.
        hparams:              Hyperparameter container.
        training_parameters:  CPU tensor [N, n_params].
        training_data:        CPU tensor [N, n_maps, 5, H, W].
        verbose:              Print progress.
        test:                 Smoke-test (3 epochs per stage).
        n_mdn_components:     MDN Gaussian components (Stage 1).

    Returns:
        (s1_val_loss, s2_val_log_prob, compressor_module, nle_module)
        All returned objects live on CPU after the call.
    """
    if verbose and rank == 0:
        print("=" * 60)
        print(f"STAGE 1: C3NN + MDN-VMIM  [world_size={world_size}]")
        print("=" * 60)

    compressor, s1_val = train_stage1_ddp(
        rank=rank, 
        local_rank=local_rank, 
        world_size=world_size, 
        hparams=hparams,
        maps_mmap=c3nn_maps_mmap,              # Use C3NN mmaps
        training_parameters=c3nn_parameters,   # Use C3NN params
        training_2pt=c3nn_2pt,                 # Use C3NN 2pt
        n_components=n_mdn_components,
        verbose=verbose,
        test=test,
    )

    if verbose and rank == 0:
        print("=" * 60)
        print(f"STAGE 2: NLE (frozen compressor)  [world_size={world_size}]")
        print("=" * 60)

    s2_val, nle_module = train_stage2_ddp(
        rank=rank, 
        local_rank=local_rank, 
        world_size=world_size, 
        hparams=hparams,
        compressor=compressor, 
        maps_mmap=nle_maps_mmap,               # Use NLE mmaps
        training_parameters=nle_parameters,    # Use NLE params
        training_2pt=nle_2pt,                  # Use NLE 2pt
        verbose=verbose,
        test=test,
    )

    # ---- Save (rank 0 only) ---------------------------------------------
    if rank == 0 and hparams.save:
        model_dir = (
            "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models/codetesting"
            if test
            else "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models"
        )
        tag = hparams.name_model() + "_two_stage_nle_ddp"
        
        # Safely move models to CPU before pickling
        compressor_cpu = compressor.cpu()
        nle_cpu        = nle_module.cpu()
        
        with open(f"{model_dir}/{tag}_compressor.pkl", "wb") as fh:
            cpk.dump(compressor_cpu, fh)
        with open(f"{model_dir}/{tag}_nle_net.pkl", "wb") as fh:
            cpk.dump(nle_cpu, fh)
        print("Models saved.")
    elif rank == 0:
        print("Not saving.")

    return s1_val, s2_val, compressor, nle_module


# =============================================================================
# __main__ — torchrun entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-stage NLE DDP training. Launch with torchrun."
    )
    parser.add_argument("-t", "--test", action="store_true",
                        help="Smoke-test mode (3 epochs per stage)")
    parser.add_argument("--mdn-components", type=int, default=8,
                        help="MDN Gaussian components for Stage 1 (default: 8)")
    parser.add_argument("--master-port", type=str, default="12355",
                        help="MASTER_PORT for DDP init (default: 12355)")
    args = parser.parse_args()

    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # <-- Extract LOCAL_RANK
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", args.master_port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(hours=2),
    )
    
    torch.cuda.set_device(local_rank) # <-- Strictly bind process to local GPU

    # Reproducibility (same seed on all ranks; DDP broadcasts rank-0 weights)
    seed = 1750
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True

    if args.test and rank == 0:
        print("Running in test mode")

    time_ini = time.time()

    # Load data on every rank (dataset fits in CPU RAM; avoids broadcast overhead)
    if rank == 0:
        print("Loading mmaps ...")

    c3nn_indices = np.load('/pscratch/sd/l/lgong/C3NN_inference/data/c3nn_training_data_param_indices.npy')
    nle_indices = np.load('/pscratch/sd/l/lgong/C3NN_inference/data/nle_training_data_param_indices.npy')
    
    c3nn_maps, c3nn_params, c3nn_2pt = load_partitioned_mmap('/pscratch/sd/l/lgong/C3NN_inference/data/c3nn_training_maps_mmap_smoothed.npy', c3nn_indices)
    nle_maps, nle_params, nle_2pt = load_partitioned_mmap('/pscratch/sd/l/lgong/C3NN_inference/data/nle_training_maps_mmap_smoothed.npy', nle_indices)

    if args.test:
        c3nn_maps = c3nn_maps[:3200]
        nle_maps = nle_maps[:3200]
        c3nn_params = c3nn_params[:3200]
        nle_params = nle_params[:3200]
        c3nn_2pt = c3nn_2pt[:3200]
        nle_2pt = nle_2pt[:3200]

    main_ddp(
        rank, local_rank, world_size,  # <-- Pass local_rank
        hparams,
        c3nn_maps, c3nn_params, c3nn_2pt,
        nle_maps, nle_params, nle_2pt,
        test=args.test,
        n_mdn_components=args.mdn_components,
    )

    cleanup_ddp()

    if rank == 0:
        print("Finished. Time elapsed:",
              datetime.timedelta(seconds=time.time() - time_ini))
