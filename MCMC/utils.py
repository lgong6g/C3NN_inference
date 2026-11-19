import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from typing import Tuple, Dict, Any, List
import sys
from pathlib import Path

from sbi.utils import BoxUniform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


n_cosmos = 2500
n_nuisance = 7
n_reals = 4

maps = np.zeros((n_cosmos, n_nuisance, n_reals, 3))
maps[:,:,:,0] = np.arange(n_cosmos)[:,None,None]
maps[:,:,:,1] = np.arange(n_nuisance)[None, :,None]
maps[:,:,:,2] = np.arange(n_reals)[None,None,:]
maps = maps.reshape(-1,3)


def convex_polygon_log_prob_torch(
    points: torch.Tensor,
    vertices: torch.Tensor,
    steepness: float = 500.0
) -> torch.Tensor:
    """Differentiable log-probability for points inside a convex polygon (GPU-compatible).

    Uses signed distance to each edge with log-sigmoid smoothing so that
    gradients flow smoothly for HMC/NUTS samplers.

    Args:
        points:    (N, 2) query points on the same device as vertices.
        vertices:  (V, 2) polygon vertices (any consistent winding order).
        steepness: sharpness of the sigmoid boundary (higher = sharper).

    Returns:
        (N,) log-probabilities (≈0 well inside, →-∞ outside).
    """
    n_v = vertices.shape[0]
    next_idx = torch.arange(1, n_v + 1, device=vertices.device) % n_v
    edges = vertices[next_idx] - vertices                       # (V, 2)
    edge_lengths = torch.norm(edges, dim=1, keepdim=True)       # (V, 1)

    # diff[i, j] = points[i] - vertices[j]
    diff = points.unsqueeze(1) - vertices.unsqueeze(0)          # (N, V, 2)

    # 2-D cross product  edge × (point − vertex)
    cross = (edges[:, 0].unsqueeze(0) * diff[:, :, 1]
           - edges[:, 1].unsqueeze(0) * diff[:, :, 0])          # (N, V)
    signed_dists = cross / edge_lengths.squeeze(-1).unsqueeze(0)

    # Determine winding from centroid so the sign convention is consistent
    centroid = vertices.mean(dim=0)
    c_diff = centroid - vertices                                # (V, 2)
    c_cross = (edges[:, 0] * c_diff[:, 1]
             - edges[:, 1] * c_diff[:, 0])                     # (V,)
    if c_cross.sum() < 0:          # CW winding → flip so "inside" = positive
        signed_dists = -signed_dists

    return F.logsigmoid(steepness * signed_dists).sum(dim=1)    # (N,)


class PolytopeBoxUniform(BoxUniform):
    """BoxUniform restricted to one or more convex polygon constraints on
    selected parameter subsets (e.g. (Om, sigma8) and (Om, w0)).

    Vertices are expected in the same space as `low`/`high` (i.e. already
    z-scored if the surrounding code is). Each polygon is converted once into
    its half-plane representation so log_prob is a single einsum per polygon.
    Sampling is rejection within the bounding box.
    """

    def __init__(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        polygon_constraints: List[Tuple[Tuple[int, ...], torch.Tensor]],
        device: str = 'cpu',
    ) -> None:
        super().__init__(low=low, high=high, device=device)
        self._device = device
        self._planes: List[Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]] = []
        for dims, verts in polygon_constraints:
            verts = verts.to(device).float()
            normals, offsets = self._half_planes(verts)
            self._planes.append((tuple(dims), normals, offsets))

    @staticmethod
    def _half_planes(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edges = torch.roll(vertices, -1, 0) - vertices                 # (M, 2)
        normals = torch.stack([edges[:, 1], -edges[:, 0]], dim=1)       # (M, 2)
        centroid = vertices.mean(0)
        outward = (normals * (vertices - centroid)).sum(-1) > 0         # (M,)
        normals = torch.where(outward.unsqueeze(1), normals, -normals)
        offsets = (normals * vertices).sum(-1)                          # (M,)
        return normals, offsets

    def _in_polytopes(self, theta: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(theta.shape[:-1], dtype=torch.bool, device=theta.device)
        for dims, normals, offsets in self._planes:
            pts = theta[..., list(dims)]                                # (..., 2)
            margin = torch.einsum('md,...d->...m', normals, pts) - offsets
            mask = mask & (margin <= 0).all(-1)
        return mask

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        base = super().log_prob(theta)
        return torch.where(
            self._in_polytopes(theta),
            base,
            torch.full_like(base, -float('inf')),
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        n = int(sample_shape.numel())
        out: List[torch.Tensor] = []
        remaining = n
        oversample = 4
        while remaining > 0:
            cand = super().sample((max(remaining * oversample, 1),))
            kept = cand[self._in_polytopes(cand)][:remaining]
            out.append(kept)
            remaining -= kept.shape[0]
        result = torch.cat(out, dim=0)[:n]
        return result.reshape(*sample_shape, self.event_shape[0])


def get_data_splits_nle_vmim(vmim: bool = False, verbose: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:

    test_inds = np.where((maps[:,1]==6) & (maps[:,2]==3))[0]

    if vmim:
        val_inds_npe = np.where((maps[:,1]==5) & (maps[:,2]==3))[0]
        val_inds_nle = np.where((maps[:,1]==4) & (maps[:,2]>=2))[0]
        train_inds_npe = np.setdiff1d(np.where((maps[:,1]>=4))[0], np.concatenate([test_inds, val_inds_npe, val_inds_nle]))
        train_inds_nle = np.setdiff1d(np.arange(maps.shape[0]), np.concatenate([test_inds, val_inds_npe, val_inds_nle, train_inds_npe]))

        all_inds_nle = np.concatenate([train_inds_nle, val_inds_nle])
        all_inds_npe = np.concatenate([train_inds_npe, val_inds_npe])

        if verbose:
            print(f"Number of training samples NLE: {len(train_inds_nle)}")
            print(f"Number of training samples NPE: {len(train_inds_npe)}")
            print(f"Number of validation samples NLE: {len(val_inds_nle)}")
            print(f"Number of validation samples NPE: {len(val_inds_npe)}")
            print(f"Number of test samples: {len(test_inds)}")
            print()

        return {'use_indices': all_inds_nle,
                'use_indices_vmim': all_inds_npe,
                'validation_indices': val_inds_nle,
                'validation_indices_vmim': val_inds_npe,
                }, test_inds

    else:
        val_inds_nle = np.concatenate([
                                        np.where((maps[:, 1] == 4) & (maps[:, 2] >= 2))[0],
                                        np.where((maps[:, 1] == 5) & (maps[:, 2] == 3))[0]
                                        ])

        train_inds_nle = np.setdiff1d(np.arange(maps.shape[0]), np.concatenate([test_inds, val_inds_nle]))

        all_inds_nle = np.concatenate([train_inds_nle, val_inds_nle])

        if verbose:
            print(f"Number of training samples: {len(train_inds_nle)}")
            print(f"Number of validation samples: {len(val_inds_nle)}")
            print(f"Number of test samples: {len(test_inds)}")
            print()

        return {'use_indices': all_inds_nle,
                'validation_indices': val_inds_nle,
                }, test_inds

def create_dataloaders(use_inds, val_inds, theta, x, validation_fraction, default_shape, training_batch_size, device):

    if use_inds is not None:
        use_inds = use_inds
    else:
        use_inds = np.arange(default_shape)

    if val_inds is not None:
        val_inds = val_inds
    else:
        val_inds = np.random.choice(
            use_inds,
            int(validation_fraction * default_shape),
            replace=False
        )

    validation_indices = torch.tensor(val_inds, device=device)
    
    # Get np.setdiff1d equivalent in PyTorch
    combined = torch.cat([torch.tensor(use_inds, device=device), validation_indices])
    uniques, counts = combined.unique(return_counts=True)
    training_indices = uniques[counts == 1]
    
    dataset_training = TensorDataset(theta[training_indices], x[training_indices])
    dataset_validation = TensorDataset(theta[validation_indices], x[validation_indices])

    train_loader_kwargs = {
        "batch_size": min(training_batch_size, training_indices.shape[0]),
        "drop_last": True,
        "shuffle": True,
    }

    val_loader_kwargs = {
        "batch_size": min(training_batch_size, validation_indices.shape[0]),
        "shuffle": False,
        "drop_last": True,
    }

    train_loader = DataLoader(dataset_training, **train_loader_kwargs)
    val_loader = DataLoader(dataset_validation, **val_loader_kwargs)

    return train_loader, val_loader

def train_epochs(
    model: torch.nn.Module,
    max_num_epochs: int,
    train_loader: "torch.utils.data.DataLoader",
    val_loader: "torch.utils.data.DataLoader",
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    stop_after_epochs: int,
    verbose: bool = True,
    mode: str = "nle",
    device: str = "cuda:0",
) -> Tuple[float, int]:

    epoch, val_loss = 0, float("Inf")
    best_model_state_dict = deepcopy(model.state_dict())
    epochs_since_last_improvement = 0
    best_val_loss = float("Inf")
    converged = False

    while epoch <= max_num_epochs and not converged:

        if verbose:
            print(f"Training {mode.upper()} epoch {epoch}...", end='\r')

     
        model.train()
        train_loss_sum = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x_batch, theta_batch = (
                batch[0].to(device),
                batch[1].to(device),
            )

            train_losses = model.loss(x_batch, condition=theta_batch)
            train_loss = torch.mean(train_losses)
            train_loss_sum += train_losses.sum().item()

            train_loss.backward()

            optimizer.step()

        epoch += 1

        # Calculate validation performance.
        model.eval()
        val_loss_sum = 0

        with torch.no_grad():
            for batch in val_loader:
                x_batch, theta_batch = (batch[0].to(device), batch[1].to(device))

                val_losses = model.loss(
                    x_batch,
                    condition=theta_batch,
                )
                val_loss_sum += val_losses.sum().item()

        scheduler.step(val_loss_sum)

        # Take mean over all validation samples.
        val_loss = val_loss_sum / (len(val_loader) * val_loader.batch_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_last_improvement = 0
            best_model_state_dict = deepcopy(model.state_dict())
        else:
            epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if epochs_since_last_improvement > stop_after_epochs - 1:
            model.load_state_dict(best_model_state_dict)
            converged = True
            if verbose:
                print(f"{mode.upper()} training converged after {epoch} epochs.")
                print(f'Best validation loss: {best_val_loss:.4f}')
                print()

    return best_val_loss, epoch

def get_hyperparams(
    param: str,
    part: str,
    hyperparam_tuning_dict: Dict[str, Any],
    trial: optuna.Trial,
    default: Any,
    log: bool = False,
    continuous: bool = False
) -> Any:

    if part == 'nle':
        hyperparam_ranges = hyperparam_tuning_dict.get(
            'hyperparam_ranges', {}
        )
        hyperparam_choices = hyperparam_tuning_dict.get(
            'hyperparam_choices', {}
        )
        fixed_params = hyperparam_tuning_dict.get('fixed_params', {})

    elif part == 'npe':
        hyperparam_ranges = hyperparam_tuning_dict.get(
            'hyperparam_npe_ranges', {}
        )
        hyperparam_choices = hyperparam_tuning_dict.get(
            'hyperparam_npe_choices', {}
        )
        fixed_params = hyperparam_tuning_dict.get('fixed_params_npe', {})

    elif part == 'embedding':
        hyperparam_ranges = hyperparam_tuning_dict.get(
            'hyperparam_embedding_ranges', {}
        )
        hyperparam_choices = hyperparam_tuning_dict.get(
            'hyperparam_embedding_choices', {}
        )
        fixed_params = hyperparam_tuning_dict.get(
            'fixed_params_embedding', {}
        )
    else:
        raise ValueError(f"Unknown option {part}")

    pname = param + '_' + part
    if param in hyperparam_ranges:
        if continuous:
            return trial.suggest_float(
                pname,
                hyperparam_ranges[param][0],
                hyperparam_ranges[param][1],
                log=log
            )
        else:
            return trial.suggest_int(
                pname,
                hyperparam_ranges[param][0],
                hyperparam_ranges[param][1]
            )
    elif param in hyperparam_choices:
        return trial.suggest_categorical(pname, hyperparam_choices[param])
    elif param in fixed_params:
        return fixed_params[param]
    else:
        return default


import torch.nn as nn

ACTIVATION_MAP = {
    'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU, 'GELU': nn.GELU,
    'ELU': nn.ELU, 'SiLU': nn.SiLU, 'Tanh': nn.Tanh,
}

def activation_to_str(act):
    if isinstance(act, str):
        return act
    return act.__name__

def str_to_activation(name: str):
    if name not in ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}. Known: {list(ACTIVATION_MAP.keys())}")
    return ACTIVATION_MAP[name]


def save_model(model: "NLE", path: str) -> None:
    hyperparams_embedding = model.hyperparams_embedding
    if hyperparams_embedding and 'activation' in hyperparams_embedding:
        hyperparams_embedding = {**hyperparams_embedding,
            'activation': activation_to_str(hyperparams_embedding['activation'])}

    checkpoint = {
        'config': {
            'order': model.order,
            'omitted_bin': model.omitted_bin,
            'moped': model.moped,
            'params': model.param_names,
            'z_score': model.z_score,
            'vmim': model.vmim,
        },
        'hyperparam_dict': model.hyperparam_dict,
        'hyperparams_embedding': hyperparams_embedding,
        'hyperparams_npe': model.hyperparams_npe,
        # VMIM-only checkpoints (best_vmim.pt) have no NLE estimator built.
        'density_estimator_state_dict': (
            model.density_estimator.state_dict()
            if model.density_estimator is not None else None
        ),
        'embedding_net_state_dict': (
            model.embedding_net.state_dict()
            if model.vmim and model.embedding_net else None
        ),
    }
    torch.save(checkpoint, path)


def load_model(path: str, device: str = "cpu") -> "NLE":
    from src.ml_methods import NLE

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint['config']

    nle = NLE(
        order=cfg['order'],
        omitted_bin=cfg['omitted_bin'],
        moped=cfg['moped'],
        params=cfg['params'],
        z_score=cfg['z_score'],
        device=device,
    )

    hyperparams_embedding = checkpoint.get('hyperparams_embedding')
    if hyperparams_embedding and 'activation' in hyperparams_embedding:
        hyperparams_embedding = {**hyperparams_embedding,
            'activation': str_to_activation(hyperparams_embedding['activation'])}

    nle.setup_model(
        vmim=cfg['vmim'],
        hyperparam_dict=checkpoint['hyperparam_dict'],
        hyperparams_embedding=hyperparams_embedding,
        hyperparams_npe=checkpoint.get('hyperparams_npe'),
    )

    nle_state = checkpoint.get('density_estimator_state_dict')
    if nle_state is not None and nle.density_estimator is not None:
        nle.density_estimator.load_state_dict(nle_state)
    if cfg['vmim'] and checkpoint.get('embedding_net_state_dict'):
        nle.embedding_net.load_state_dict(checkpoint['embedding_net_state_dict'])

    return nle
