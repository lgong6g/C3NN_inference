"""Stage 2: Train NLE (Neural Spline Flow) on frozen C3NN summaries.

Runs on a single GPU.  The compressed summaries and matching parameter array
must be produced first by train_c3nn_ddp.py.

Usage:
    python train_nle.py \\
        --summaries /path/to/tag_nle_summaries.npy \\
        --params    /path/to/tag_nle_params.npy    \\
        [--resume]  [--test]

Outputs:
    {model_dir}/{tag}_nle_net.pkl           -- final NLE network (cloudpickle)
    {checkpoint_dir}/nle_epoch_{NNNN}.pt    -- periodic training checkpoints
"""

import sys
sys.path.append('./src')

import os
import glob
import argparse
import time
import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cloudpickle as cpk

from torch.utils.data import DataLoader, TensorDataset

from sbi.neural_nets import likelihood_nn

from hyperparameters import hparams

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path of the highest-numbered NLE checkpoint, or None."""
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "nle_epoch_*.pt")))
    return files[-1] if files else None


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    nle_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_val_lp: float,
    best_state: dict | None,
    patience_ctr: int,
) -> None:
    """Persist an NLE training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"nle_epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": nle_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_lp": best_val_lp,
            "best_state": best_state,
            "patience_ctr": patience_ctr,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# NLE training
# ---------------------------------------------------------------------------

def train_nle(
    summaries: np.ndarray,
    params: np.ndarray,
    hparams,
    device: torch.device,
    checkpoint_dir: str,
    max_epochs: int = 300,
    early_stop_patience: int = 10,
    checkpoint_every: int = 20,
    resume: bool = False,
    verbose: bool = True,
    test: bool = False,
) -> tuple[float, nn.Module]:
    """Train an NSF likelihood network on pre-computed summaries.

    Args:
        summaries:   float32 array [N, out_dim] — compressed C3NN summaries.
        params:      float32 array [N, n_params] — cosmological parameters.
        hparams:     Hyperparameter container.
        device:      Target CUDA device.
        checkpoint_dir: Directory for reading/writing checkpoints.
        max_epochs:  Maximum training epochs.
        early_stop_patience: Early-stopping patience.
        checkpoint_every: Save a checkpoint every this many epochs.
        resume:      If True, resume from the latest checkpoint in checkpoint_dir.
        verbose:     Print training progress.
        test:        Smoke-test mode (3 epochs).

    Returns:
        (best_val_log_prob, nle_net) where nle_net contains the best weights.
    """
    summaries_t = torch.tensor(summaries, dtype=torch.float32)
    params_t    = torch.tensor(params,    dtype=torch.float32)

    # ---- Build NSF network -------------------------------------------------
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
        batch_theta=params_t,
        batch_x=summaries_t,
    ).to(device)

    optimizer = optim.Adam(nle_net.parameters(), lr=hparams.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20
    )

    # ---- Train / val split (80 / 20) ---------------------------------------
    N         = summaries_t.shape[0]
    n_val     = max(1, int(0.2 * N))
    x_tr      = summaries_t[: N - n_val].to(device)
    t_tr      = params_t[: N - n_val].to(device)
    x_val     = summaries_t[N - n_val :].to(device)
    t_val     = params_t[N - n_val :].to(device)

    train_loader = DataLoader(
        TensorDataset(t_tr, x_tr),
        batch_size=hparams.flow_training_batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ---- Resume from checkpoint (if requested) -----------------------------
    start_epoch   = 0
    best_val_lp   = float("-inf")
    best_state    = None
    patience_ctr  = 0

    if resume:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            nle_net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch  = ckpt["epoch"] + 1
            best_val_lp  = ckpt["best_val_lp"]
            best_state   = ckpt.get("best_state")
            patience_ctr = ckpt["patience_ctr"]
            if verbose:
                print(f"Resumed from checkpoint: {ckpt_path}  (epoch {start_epoch})")
        else:
            if verbose:
                print("--resume requested but no checkpoint found; starting fresh.")

    # ---- Training loop -----------------------------------------------------
    n_epochs = 3 if test else max_epochs

    for epoch in range(start_epoch, n_epochs):
        nle_net.train()
        epoch_loss = 0.0
        n_batches  = 0
        for theta_b, x_b in train_loader:
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
                x_val.unsqueeze(0), condition=t_val
            )[0].mean().item()

        scheduler.step(val_lp)

        avg_train = epoch_loss / max(1, n_batches)
        if verbose and (epoch % 10 == 0 or epoch < 5):
            print(
                f"Stage 2 | Epoch {epoch:4d} | "
                f"train log-prob {-avg_train:.4f} | val log-prob {val_lp:.4f}"
            )

        # Track best
        if val_lp > best_val_lp:
            best_val_lp  = val_lp
            best_state   = {k: v.cpu().clone() for k, v in nle_net.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience and not test:
                if verbose:
                    print(
                        f"Stage 2: early stopping at epoch {epoch} "
                        f"(best val log-prob {best_val_lp:.4f})"
                    )
                save_checkpoint(
                    checkpoint_dir, epoch, nle_net, optimizer, scheduler,
                    best_val_lp, best_state, patience_ctr,
                )
                break

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir, epoch, nle_net, optimizer, scheduler,
                best_val_lp, best_state, patience_ctr,
            )

    # Restore best weights
    if best_state is not None:
        nle_net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if verbose:
        print(f"Stage 2 complete. Best val log-prob: {best_val_lp:.4f}")

    return best_val_lp, nle_net


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Train NLE on pre-computed C3NN summaries (single GPU)."
    )
    parser.add_argument(
        "--summaries", type=str, required=True,
        help="Path to .npy file of C3NN summaries [N, out_dim] from train_c3nn_ddp.py",
    )
    parser.add_argument(
        "--params", type=str, required=True,
        help="Path to .npy file of cosmological parameters [N, n_params] from train_c3nn_ddp.py",
    )
    parser.add_argument("-t", "--test", action="store_true",
                        help="Smoke-test mode (3 epochs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest NLE checkpoint")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    seed = 1750
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_dir = (
        "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models/codetesting"
        if args.test
        else "/pscratch/sd/l/lgong/C3NN_inference/C3NN_Models"
    )
    checkpoint_dir = os.path.join(model_dir, "checkpoints", "nle")

    time_ini = time.time()

    print(f"Loading summaries from: {args.summaries}")
    summaries = np.load(args.summaries)
    print(f"Loading params from:    {args.params}")
    params = np.load(args.params)
    print(f"  summaries shape: {summaries.shape}, params shape: {params.shape}")

    # Drop rows with non-finite summaries (corrupted Stage 1 outputs)
    finite_mask = np.all(np.isfinite(summaries), axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"WARNING: dropping {n_bad} rows with NaN/Inf summaries " 
              f"(indices: {np.where(~finite_mask)[0].tolist()})")
        summaries = summaries[finite_mask]
        params    = params[finite_mask]

    if args.test:
        summaries = summaries[:3200]
        params    = params[:3200]
        print("Test mode: using first 3200 samples")

    print("=" * 60)
    print("STAGE 2: NLE training (single GPU)")
    print("=" * 60)

    best_val_lp, nle_net = train_nle(
        summaries=summaries,
        params=params,
        hparams=hparams,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=20,
        resume=args.resume,
        test=args.test,
    )

    if hparams.save:
        os.makedirs(model_dir, exist_ok=True)
        tag      = hparams.name_model() + "_nle_moped_separate_5coeffs_stage2"
        out_path = os.path.join(model_dir, f"{tag}_net.pkl")
        with open(out_path, "wb") as fh:
            cpk.dump(nle_net.cpu(), fh)
        print(f"NLE model saved → {out_path}")
    else:
        print("hparams.save=False — skipping file output.")

    elapsed = datetime.timedelta(seconds=time.time() - time_ini)
    print(f"Stage 2 finished. Elapsed: {elapsed}")
