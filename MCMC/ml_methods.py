import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import warnings
warnings.filterwarnings("ignore")

import sys
import math
import numpy as np
from typing import Optional, Dict, Any, List
from numpy.typing import NDArray

from pathlib import Path
import emcee
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sbi.neural_nets.net_builders import build_mdn, build_zuko_gf
from sbi.neural_nets.net_builders import build_zuko_gf as build_gf
from sbi.neural_nets.net_builders import build_zuko_maf as build_maf
from sbi.neural_nets.net_builders import build_zuko_nsf as build_nsf
from sbi.inference import likelihood_estimator_based_potential

try:
    from tqdm import trange
except ImportError:
    trange = None


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import *
from src.utils import train_epochs, get_hyperparams, create_dataloaders, get_data_splits_nle_vmim, save_model, load_model, PolytopeBoxUniform

default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
if default_device == "cpu":
    torch.serialization.register_package(
        0, lambda x: x.device.type, lambda x, _: x.cpu()
    )


class FCNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 1440,
        hidden_features: int = 1024,
        output_dim: int = 18,
        num_blocks: int = 2,
        dropout: float = 0.1,
        norm_type: Optional[str] = 'layer',
        activation: type[nn.Module] = nn.ReLU,
        device: str = default_device
    ) -> None:

        super().__init__()
        
        self.block_list = nn.ModuleList()
        
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(input_dim, device=device)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(input_dim, device=device)
        elif norm_type is not None:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        
        for _ in range(num_blocks):
            in_features = input_dim if _ == 0 else hidden_features
            block = nn.Sequential(
                nn.Linear(in_features, hidden_features, device=device),
                activation(inplace=True),
                nn.Dropout(dropout),
            )
            self.block_list.append(block)
        
        self.head = nn.Linear(hidden_features, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "norm"):
            x = self.norm(x)
        for block in self.block_list:
            x = block(x)
        return self.head(x)


class NDE:

    # Process-wide cache so 800 HPT trials don't re-read the same .npy/.npz
    # files. Tensors are never mutated in place downstream (z-scoring rebinds
    # self.theta / self.x_grid to fresh tensors), so sharing is safe.
    _data_cache: Dict[tuple, Dict[str, Any]] = {}

    @classmethod
    def _load_data(
        cls,
        order: str,
        omitted_bin: Optional[int],
        moped: bool,
        param_names: List[str],
        device: str,
    ) -> Dict[str, Any]:
        cache_key = (order, omitted_bin, moped, tuple(param_names), str(device))
        if cache_key in cls._data_cache:
            return cls._data_cache[cache_key]

        theta_dict = dict(np.load(DATA_DIR / "params" / "theta_dict.npz"))
        theta_array = np.array([theta_dict[par] for par in param_names]).T
        theta = torch.tensor(theta_array, device=device).float()

        theta_dict_fiducial = dict(
            np.load(DATA_DIR / "params" / "theta_dict_fiducial.npz")
        )
        theta_fiducial = np.array(
            [theta_dict_fiducial[par] for par in param_names]
        )

        summary_dir = DATA_DIR / "summaries"
        if moped:
            fname_grid = f"moped_coeffs_{order}_grid"
            fname_fid = f"moped_coeffs_{order}_cov"
        else:
            fname_grid = f"dv_{order}_grid"
            fname_fid = f"dv_{order}_cov"
        if omitted_bin is not None:
            fname_grid += f"_no_bin{omitted_bin + 1}"
            fname_fid += f"_no_bin{omitted_bin + 1}"
        fname_grid += ".npy"
        fname_fid += ".npy"

        x_grid = torch.tensor(np.load(summary_dir / fname_grid), device=device).float()
        x_fid = torch.tensor(np.load(summary_dir / fname_fid), device=device).float()

        data = {
            'theta': theta,
            'theta_fiducial': theta_fiducial,
            'x_grid': x_grid,
            'x_fid': x_fid,
        }
        cls._data_cache[cache_key] = data
        return data

    def __init__(
        self,
        order: str,
        omitted_bin: Optional[int] = None,
        moped: bool = False,
        verbose: bool = False,
        params: Optional[List[str]] = None,
        z_score: bool = False,
        device: str = default_device
    ) -> None:

        self.device = device
        self.order = order
        self.omitted_bin = omitted_bin
        self.moped = moped
        self.verbose = verbose
        self.z_score = z_score
        if params is not None:
            self.param_names = params
        else:
            self.param_names = [
                "Om", "Ob", "H0", "ns", "sigma8", "w0",
                "A_ia", "bary_Mc", "bary_nu"
            ]

        data = self._load_data(
            self.order, self.omitted_bin, self.moped,
            self.param_names, self.device,
        )
        self.theta = data['theta']
        self.theta_fiducial = data['theta_fiducial']
        self.x_grid = data['x_grid']
        self.x_fid = data['x_fid']
        
        if self.z_score:
            self.theta_mean = self.theta.mean(dim=0, keepdim=True)
            self.theta_std = self.theta.std(dim=0, keepdim=True)
            self.theta = (self.theta - self.theta_mean) / self.theta_std

            self.x_mean = self.x_grid.mean(dim=0, keepdim=True)
            self.x_std = self.x_grid.std(dim=0, keepdim=True)
            self.x_grid = (self.x_grid - self.x_mean) / self.x_std
        
        self.num_simulations = self.theta.shape[0]
        self.parameter_dim = self.theta.shape[1]
        self.data_dim = self.x_grid.shape[1]

        polygon_constraints = []
        if "Om" in self.param_names and "sigma8" in self.param_names:
            om_ind = self.param_names.index("Om")
            sig8_ind = self.param_names.index("sigma8")
            poly_om_sig8 = torch.tensor([
                [0.1, 1.4], [0.2, 1.4], [0.5, 0.65],
                [0.5, 0.4], [0.3, 0.4], [0.1, 0.9],
            ], device=self.device, dtype=torch.float32)
            if self.z_score:
                poly_om_sig8[:, 0] = (poly_om_sig8[:, 0] - self.theta_mean[0, om_ind]) / self.theta_std[0, om_ind]
                poly_om_sig8[:, 1] = (poly_om_sig8[:, 1] - self.theta_mean[0, sig8_ind]) / self.theta_std[0, sig8_ind]
            polygon_constraints.append(((om_ind, sig8_ind), poly_om_sig8))

        if "Om" in self.param_names and "w0" in self.param_names:
            om_ind = self.param_names.index("Om")
            w0_ind = self.param_names.index("w0")
            poly_om_w0 = torch.tensor([
                [0.1, -1.1], [0.5, -2.0], [0.5, -0.33], [0.1, -0.33],
            ], device=self.device, dtype=torch.float32)
            if self.z_score:
                poly_om_w0[:, 0] = (poly_om_w0[:, 0] - self.theta_mean[0, om_ind]) / self.theta_std[0, om_ind]
                poly_om_w0[:, 1] = (poly_om_w0[:, 1] - self.theta_mean[0, w0_ind]) / self.theta_std[0, w0_ind]
            polygon_constraints.append(((om_ind, w0_ind), poly_om_w0))

        self.prior = PolytopeBoxUniform(
            low=self.theta.min(dim=0).values,
            high=self.theta.max(dim=0).values,
            polygon_constraints=polygon_constraints,
            device=self.device,
        )

    def _get_buf(
        self,
        name: str,
        shape,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a cached GPU buffer; reallocate if the shape/dtype changed.

        Long sampling loops reuse the same chain/log_prob buffers across
        calls so PyTorch doesn't churn through (re-)allocations of
        hundreds of MB per posterior.
        """
        shape = tuple(shape)
        buf = getattr(self, name, None)
        if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
            buf = torch.empty(shape, device=self.device, dtype=dtype)
            setattr(self, name, buf)
        return buf


class NLE(NDE):

    def __init__(
        self,
        order: str,
        omitted_bin: Optional[int] = None,
        moped: bool = False,
        verbose: bool = False,
        params: Optional[List[str]] = None,
        z_score: bool = False,
        device: str = default_device
    ) -> None:
        super().__init__(
            order, omitted_bin, moped, verbose, params, z_score, device
        )
        self.x_fid_embedded = None
        self.x_grid_embedded = None
        self.use_indices_vmim = None
        self.validation_indices = None
        self.use_indices = None
        self.validation_indices_vmim = None
        self.density_estimator = None
        self.embedding_net = None
        self.density_estimator_vmim = None
        self.vmim = False
        self.hyperparam_dict = None
        self.hyperparams_embedding = None
        self.hyperparams_npe = None
        self.nle_epochs_trained = 0
        self.vmim_epochs_trained = 0

    def setup_model(
        self,
        vmim: bool = False,
        hyperparam_dict: Optional[Dict[str, Any]] = None,
        hyperparams_embedding: Optional[Dict[str, Any]] = None,
        hyperparams_npe: Optional[Dict[str, Any]] = None,
        cached_embedding_net: Optional[nn.Module] = None,
        cached_x_grid_embedded: Optional[torch.Tensor] = None,
        cached_x_fid_embedded: Optional[torch.Tensor] = None,
    ) -> None:

        self.vmim = vmim
        self.hyperparams_embedding = None
        self.hyperparams_npe = None

        using_cached_embedding = (
            vmim
            and cached_embedding_net is not None
            and cached_x_grid_embedded is not None
            and cached_x_fid_embedded is not None
        )

        if using_cached_embedding:
            self.embedding_net = cached_embedding_net
            self.x_grid_embedded = cached_x_grid_embedded
            self.x_fid_embedded = cached_x_fid_embedded
            self.hyperparams_embedding = hyperparams_embedding
            self.hyperparams_npe = hyperparams_npe
            latent_dim = self.x_grid_embedded.shape[1]

        elif vmim:
            if hyperparams_embedding is not None:
                hidden_features = hyperparams_embedding.get(
                    'hidden_features', 128
                )
                num_blocks = hyperparams_embedding.get('num_blocks', 2)
                dropout = hyperparams_embedding.get('dropout', 0.1)
                norm_type = hyperparams_embedding.get('norm_type', None)
                activation = hyperparams_embedding.get('activation', nn.ReLU)
                latent_dim = hyperparams_embedding.get('latent_dim', 2 * self.parameter_dim)

            else:
                hidden_features = 128
                num_blocks = 2
                dropout = 0.1
                norm_type = None
                activation = nn.ReLU
                latent_dim = 2 * self.parameter_dim

            self.hyperparams_embedding = {
                'hidden_features': hidden_features, 'num_blocks': num_blocks,
                'dropout': dropout, 'norm_type': norm_type,
                'activation': activation, 'latent_dim': latent_dim,
            }

            self.embedding_net = FCNet(
                self.data_dim,
                hidden_features=hidden_features,
                output_dim=latent_dim,
                num_blocks=num_blocks,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation,
                device=self.device
            )

            if hyperparams_npe is not None:
                model = hyperparams_npe.get('model', 'mdn')
                hidden_features = hyperparams_npe.get('hidden_features', 50)
                num_components = hyperparams_npe.get('num_components', 1)
                num_blocks = hyperparams_npe.get('num_blocks', 2)
                num_transforms = hyperparams_npe.get('num_transforms', 5)
                hidden_layers_spline_context = hyperparams_npe.get('hidden_layers_spline_context', 1)
                num_bins = hyperparams_npe.get('num_bins', 10)
                batch_norm = hyperparams_npe.get('batch_norm', False)
                dropout = hyperparams_npe.get('dropout', 0.0)

            else:
                model = 'mdn'
                num_components = 1
                num_transforms = 5
                hidden_features = 50
                num_blocks = 2
                num_bins = 10
                hidden_layers_spline_context = 1
                batch_norm = False
                dropout = 0.0

            self.hyperparams_npe = {
                'model': model, 'hidden_features': hidden_features,
                'num_components': num_components, 'num_blocks': num_blocks,
                'num_transforms': num_transforms,
                'hidden_layers_spline_context': hidden_layers_spline_context,
                'num_bins': num_bins, 'batch_norm': batch_norm,
                'dropout': dropout,
            }

            if model == 'mdn':
                self.density_estimator_vmim = build_mdn(self.theta,
                                                   self.x_grid,
                                                   z_score_x='none',
                                                   z_score_y='none',
                                                   hidden_features=hidden_features,
                                                   num_blocks=num_blocks,
                                                   num_components=num_components,
                                                   dropout_probability=dropout,
                                                   embedding_net=self.embedding_net,
                                                   device=self.device)
            elif model in ('maf', 'gf', 'nsf'):
                hidden_features_list = [hidden_features] * num_blocks
                if model == 'maf':
                    self.density_estimator_vmim = build_maf(self.theta,
                                                       self.x_grid,
                                                       z_score_x='none',
                                                       z_score_y='none',
                                                       hidden_features=hidden_features_list,
                                                       num_transforms=num_transforms,
                                                       embedding_net=self.embedding_net)
                elif model == 'gf':
                    self.density_estimator_vmim = build_gf(self.theta,
                                                       self.x_grid,
                                                       z_score_x='none',
                                                       z_score_y='none',
                                                       hidden_features=hidden_features_list,
                                                       num_transforms=num_transforms,
                                                       embedding_net=self.embedding_net)
                else:
                    self.density_estimator_vmim = build_nsf(self.theta,
                                                       self.x_grid,
                                                       z_score_x='none',
                                                       z_score_y='none',
                                                       hidden_features=hidden_features_list,
                                                       num_transforms=num_transforms,
                                                       num_bins=num_bins,
                                                       embedding_net=self.embedding_net)
            else:
                raise NotImplementedError(
                    f"Model {model} not implemented for VMIM-NPE."
                )

            self.density_estimator_vmim.to(self.device)

        # Non-cached VMIM has a fresh embedding net whose outputs are junk —
        # any z-score buffers we'd compute here would be wrong. Defer NLE
        # construction unless we're rebuilding from saved hyperparams
        # (load_model overwrites the buffers via state_dict afterwards).
        if using_cached_embedding:
            nle_batch_x = self.x_grid_embedded
            z_score_x_nle = 'independent'
        elif vmim:
            if hyperparam_dict is None:
                self.density_estimator = None
                return
            with torch.inference_mode():
                nle_batch_x = self.embedding_net(self.x_grid).detach()
            z_score_x_nle = 'independent'
        else:
            # self.x_grid is already z-scored when self.z_score=True; asking
            # sbi to z-score again would just stack two identity normalizers.
            nle_batch_x = self.x_grid
            z_score_x_nle = 'none' if self.z_score else 'independent'

        nle_batch_y = self.theta

        if hyperparam_dict is not None:
            model = hyperparam_dict.get('model', 'maf')
            hidden_features = hyperparam_dict.get('hidden_features', 128)
            num_blocks = hyperparam_dict.get('num_blocks', 2)
            num_components = hyperparam_dict.get('num_components', 5)
            num_transforms = hyperparam_dict.get('num_transforms', 5)
            hidden_layers_spline_context = hyperparam_dict.get('hidden_layers_spline_context', 1)
            num_bins = hyperparam_dict.get('num_bins', 10)
            batch_norm = hyperparam_dict.get('batch_norm', False)
            dropout = hyperparam_dict.get('dropout', 0.0)
        else:
            model = 'mdn'
            num_components = 1
            num_transforms = 5
            hidden_features = 50
            num_blocks = 2
            num_bins = 10
            hidden_layers_spline_context = 1
            batch_norm = False
            dropout = 0.0

        self.hyperparam_dict = {
            'model': model, 'hidden_features': hidden_features,
            'num_blocks': num_blocks, 'num_components': num_components,
            'num_transforms': num_transforms,
            'hidden_layers_spline_context': hidden_layers_spline_context,
            'num_bins': num_bins, 'batch_norm': batch_norm,
            'dropout': dropout,
        }

        if model == 'mdn':
            self.density_estimator = build_mdn(nle_batch_x,
                                               nle_batch_y,
                                               z_score_x=z_score_x_nle,
                                               z_score_y='none',
                                               hidden_features=hidden_features,
                                               num_blocks=num_blocks,
                                               num_components=num_components,
                                               dropout_probability=dropout,
                                               device=self.device)
        elif model in ('maf', 'gf', 'nsf'):
            hidden_features_list = [hidden_features] * num_blocks
            if model == 'maf':
                self.density_estimator = build_maf(nle_batch_x,
                                                   nle_batch_y,
                                                   z_score_x=z_score_x_nle,
                                                   z_score_y='none',
                                                   hidden_features=hidden_features_list,
                                                   num_transforms=num_transforms)
            elif model == 'gf':
                self.density_estimator = build_gf(nle_batch_x,
                                                  nle_batch_y,
                                                  z_score_x=z_score_x_nle,
                                                  z_score_y='none',
                                                  hidden_features=hidden_features_list,
                                                  num_transforms=num_transforms)
            else:
                self.density_estimator = build_nsf(nle_batch_x,
                                                   nle_batch_y,
                                                   z_score_x=z_score_x_nle,
                                                   z_score_y='none',
                                                   hidden_features=hidden_features_list,
                                                   num_transforms=num_transforms,
                                                   num_bins=num_bins)
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for NLE."
            )
        self.density_estimator.to(self.device)
        
    def train(
        self,
        training_params: Optional[Dict[str, Any]],
        training_params_vmim: Optional[Dict[str, Any]] = None,
        use_indices: Optional[NDArray] = None,
        use_indices_vmim: Optional[NDArray] = None,
        validation_indices: Optional[NDArray] = None,
        validation_indices_vmim: Optional[NDArray] = None
    ) -> float:

        if self.x_grid_embedded is None:
            if self.vmim:
                _ = self.train_vmim_summary(
                    training_params_vmim=training_params_vmim,
                    use_indices_vmim=use_indices_vmim,
                    validation_indices_vmim=validation_indices_vmim
                )
            else:
                self.x_grid_embedded = self.x_grid
                self.x_fid_embedded = self.x_fid

        if training_params is not None:
            training_batch_size = training_params.get(
                'training_batch_size', 200
            )
            learning_rate = training_params.get('learning_rate', 1e-3)
            validation_fraction = training_params.get('validation_fraction', 0.1)
            stop_after_epochs = training_params.get('stop_after_epochs', 20)
            max_num_epochs = training_params.get('max_num_epochs', 1e9)
            factor = training_params.get('factor', np.sqrt(0.1))
            patience = training_params.get('patience', 5)
        else:
            training_batch_size = 200
            learning_rate = 1e-3
            validation_fraction = 0.1
            stop_after_epochs = 20
            max_num_epochs = 1e9
            factor = np.sqrt(0.1)
            patience = 5

        train_loader, val_loader = create_dataloaders(use_indices,
                                                      validation_indices,
                                                      self.x_grid_embedded,
                                                      self.theta,
                                                      validation_fraction,
                                                      self.x_grid_embedded.shape[0],
                                                      training_batch_size,
                                                      self.device)

        optimizer = Adam(list(self.density_estimator.parameters()), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

        best_val_loss, n_epochs = train_epochs(self.density_estimator,
                                     max_num_epochs,
                                     train_loader,
                                     val_loader,
                                     optimizer,
                                     scheduler,
                                     stop_after_epochs,
                                     mode = "nle",
                                     verbose=self.verbose)
        self.nle_epochs_trained = n_epochs

        return best_val_loss

    def train_vmim_summary(
        self,
        training_params_vmim: Optional[Dict[str, Any]] = None,
        use_indices_vmim: Optional[NDArray] = None,
        validation_indices_vmim: Optional[NDArray] = None
    ) -> float:

        if not self.vmim:
            self.x_grid_embedded = self.x_grid
            self.x_fid_embedded = self.x_fid
            return float("nan")

        if training_params_vmim is not None:
            training_batch_size = training_params_vmim.get('training_batch_size', 200)
            learning_rate = training_params_vmim.get('learning_rate', 1e-3)
            validation_fraction = training_params_vmim.get('validation_fraction', 0.1)
            stop_after_epochs = training_params_vmim.get('stop_after_epochs', 20)
            max_num_epochs = training_params_vmim.get('max_num_epochs', 1e9)
            patience = training_params_vmim.get('patience', 5)
            factor = training_params_vmim.get('factor', np.sqrt(0.1))
        else:
            training_batch_size = 200
            learning_rate = 1e-3
            validation_fraction = 0.1
            stop_after_epochs = 20
            max_num_epochs = 1e9
            factor = np.sqrt(0.1)
            patience = 5

        train_loader_vmim, val_loader_vmim = create_dataloaders(use_indices_vmim,
                                                                validation_indices_vmim,
                                                                self.theta,
                                                                self.x_grid,
                                                                validation_fraction,
                                                                self.x_grid.shape[0],
                                                                training_batch_size,
                                                                self.device)

        optimizer_npe = Adam(list(self.density_estimator_vmim.parameters()), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer_npe, mode='min', factor=factor, patience=patience)

        best_val_loss, n_epochs = train_epochs(self.density_estimator_vmim,
                                     max_num_epochs,
                                     train_loader_vmim,
                                     val_loader_vmim,
                                     optimizer_npe,
                                     scheduler,
                                     stop_after_epochs,
                                     mode = "npe",
                                     verbose=self.verbose)
        self.vmim_epochs_trained = n_epochs

        self.embedding_net.eval()

        with torch.inference_mode():
            self.x_grid_embedded = self.embedding_net(self.x_grid).detach()
            self.x_fid_embedded = self.embedding_net(self.x_fid).detach()

        return best_val_loss

    def get_posterior(
        self,
        x_obs: torch.Tensor,
        theta_0: NDArray,
        mcmc_params: Optional[Dict[str, Any]] = None,
        return_log_prob: bool = False
    ) -> NDArray:

        if self.z_score:
            x_obs = (x_obs - self.x_mean) / self.x_std

        if self.vmim:
            x_obs_embedded = self.embedding_net(x_obs).detach()
            potential_fn, _ = likelihood_estimator_based_potential(
                self.density_estimator, self.prior, x_o=x_obs_embedded
            )
        else:
            potential_fn, _ = likelihood_estimator_based_potential(
                self.density_estimator, self.prior, x_o=x_obs
            )

        if mcmc_params is not None:
            nwalkers = mcmc_params.get('nwalkers', 1000)
            nsteps = mcmc_params.get('nsteps', 1000)
            discard = mcmc_params.get('discard', 2000)
            thin = mcmc_params.get('thin', 3)
            progress = mcmc_params.get('progress', True)
            flat = mcmc_params.get('flat', True)
        else:
            nwalkers = 1000
            nsteps = 1000
            discard = 2000
            thin = 3
            progress = True
            flat = True

        tot_steps = nsteps * thin + discard


        def log_prob(args):

            if self.z_score:
                args = (
                        (args - self.theta_mean.cpu().numpy())
                        / self.theta_std.cpu().numpy()
                )

            return np.array(
                potential_fn(
                    torch.tensor(args).float().to(default_device)
                ).tolist()
            )

        sampler = emcee.EnsembleSampler(
            nwalkers, self.parameter_dim, log_prob, vectorize=True
        )
        sampler.run_mcmc(theta_0, tot_steps, progress=progress)
        samples = sampler.get_chain(discard=discard, thin=thin, flat=flat)

        if return_log_prob:
            log_probs = sampler.get_log_prob(
                discard=discard, thin=thin, flat=flat
            )
            return samples, log_probs

        return samples

    def get_posterior_gpu(
        self,
        x_obs: torch.Tensor,
        theta_0: Optional[NDArray] = None,
        mcmc_params: Optional[Dict[str, Any]] = None,
        return_log_prob: bool = False
    ) -> NDArray:
        """GPU-accelerated posterior sampling using affine-invariant ensemble MCMC.

        Implements the stretch move (Goodman & Weare 2010) entirely on GPU.
        All walkers in each complementary half are updated in parallel,
        avoiding the CPU-GPU data copies of the emcee-based method.

        mcmc_params keys (all optional):
            nwalkers     (int):   number of walkers              [1000]
            nsteps       (int):   number of recorded steps       [1000]
            discard      (int):   burn-in steps to discard      [10000]
            thin         (int):   thinning interval                 [3]
            stretch_a  (float):   stretch move scale parameter    [2.0]
            progress    (bool):   show progress bar              [True]
        """
        if self.z_score:
            x_obs = (x_obs - self.x_mean) / self.x_std

        if self.vmim:
            with torch.inference_mode():
                x_obs_embedded = self.embedding_net(x_obs).detach()
            potential_fn, _ = likelihood_estimator_based_potential(
                self.density_estimator, self.prior, x_o=x_obs_embedded
            )
        else:
            potential_fn, _ = likelihood_estimator_based_potential(
                self.density_estimator, self.prior, x_o=x_obs
            )

        if mcmc_params is not None:
            nwalkers = mcmc_params.get('nwalkers', 1000)
            nsteps = mcmc_params.get('nsteps', 1000)
            discard = mcmc_params.get('discard', 2000)
            thin = mcmc_params.get('thin', 3)
            stretch_a = mcmc_params.get('stretch_a', 2.0)
            progress = mcmc_params.get('progress', True)
        else:
            nwalkers = 1000
            nsteps = 1000
            discard = 2000
            thin = 3
            stretch_a = 2.0
            progress = True

        tot_steps = nsteps * thin + discard
        ndim = self.parameter_dim

        # Size the full-chain buffer against actual free GPU memory rather
        # than a fixed 2 GB cap. Reuse an already-allocated buffer when its
        # shape matches so loops of sequential calls don't realloc each time.
        # Allocated OUTSIDE inference_mode so the buffers stay regular
        # tensors — writes from inside the inference_mode block target
        # them in-place without hitting the inference-tensor restriction.
        chain_bytes_needed = tot_steps * nwalkers * ndim * 4
        existing = getattr(self, '_chain_buf', None)
        if existing is not None and existing.shape == (tot_steps, nwalkers, ndim):
            store_all = True
        elif existing is not None and existing.shape == (nsteps, nwalkers, ndim):
            store_all = False
        else:
            if 'cuda' in str(self.device):
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                max_chain_bytes = free_bytes // 2
            else:
                max_chain_bytes = 2**31
            store_all = chain_bytes_needed < max_chain_bytes

        T = tot_steps if store_all else nsteps
        chain = self._get_buf('_chain_buf', (T, nwalkers, ndim))
        if return_log_prob:
            lp_chain = self._get_buf('_lp_buf', (T, nwalkers))

        step_range = range(tot_steps)
        if progress and trange is not None:
            step_range = trange(tot_steps, desc="GPU ensemble MCMC")

        def log_prob_batch(theta_batch):
            return potential_fn(theta_batch)

        kept = 0
        with torch.inference_mode():
            # Initialise walker positions on GPU. self.prior is constructed
            # from z-scored theta, so its samples are already in z-scored
            # space — only the user-provided theta_0 needs conversion.
            if theta_0 is not None:
                walkers = torch.as_tensor(
                    theta_0, device=self.device, dtype=torch.float32
                )
                if self.z_score:
                    walkers = (walkers - self.theta_mean) / self.theta_std
            else:
                walkers = self.prior.sample((nwalkers,)).to(self.device)

            log_probs = log_prob_batch(walkers)                   # (nwalkers,)

            half = nwalkers // 2
            idx_s0 = torch.arange(0, half, device=self.device)
            idx_s1 = torch.arange(half, nwalkers, device=self.device)

            for step in step_range:
                for active_idx, comp_idx in [(idx_s0, idx_s1), (idx_s1, idx_s0)]:
                    n_active = active_idx.shape[0]
                    n_comp = comp_idx.shape[0]

                    j = torch.randint(n_comp, (n_active,), device=self.device)
                    partners = walkers[comp_idx[j]]

                    u = torch.rand(n_active, device=self.device)
                    z = ((stretch_a - 1.0) * u + 1.0) ** 2 / stretch_a

                    proposals = partners + z.unsqueeze(1) * (
                        walkers[active_idx] - partners
                    )

                    lp_prop = log_prob_batch(proposals)

                    log_accept = (ndim - 1) * torch.log(z) + lp_prop - log_probs[active_idx]
                    accept = torch.log(
                        torch.rand(n_active, device=self.device)
                    ) < log_accept

                    walkers[active_idx[accept]] = proposals[accept]
                    log_probs[active_idx[accept]] = lp_prop[accept]

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

        # Outside inference_mode: views/ops on the regular ``chain`` buffer
        # yield regular tensors that can be returned safely.
        if store_all:
            samples = chain[discard::thin].reshape(-1, ndim)
            if return_log_prob:
                lp_samples = lp_chain[discard::thin].reshape(-1)
        else:
            samples = chain[:kept].reshape(-1, ndim)
            if return_log_prob:
                lp_samples = lp_chain[:kept].reshape(-1)

        if self.z_score:
            samples = samples * self.theta_std + self.theta_mean

        if return_log_prob:
            return samples.cpu().numpy(), lp_samples.cpu().numpy()

        return samples.cpu().numpy()

    def get_posterior_gpu_batched(
        self,
        x_obs: torch.Tensor,
        theta_0: Optional[Any] = None,
        mcmc_params: Optional[Dict[str, Any]] = None,
        return_log_prob: bool = False,
    ):
        """Run B affine-invariant ensemble chains in parallel on the GPU.

        Each of the ``B = x_obs.shape[0]`` rows is a separate observation;
        all ``B * nwalkers`` walkers advance with one density-estimator
        forward per step (per half-update), so throughput grows roughly
        linearly with B until GPU memory or arithmetic saturates.

        Walker layout in the flat ``M = B * nwalkers`` arrays is
        block-major: walkers ``[b*nwalkers : (b+1)*nwalkers]`` belong to
        observation ``b``. The stretch move's complementary partners are
        drawn from the *same block* so the chains stay independent.

        Parameters
        ----------
        x_obs : Tensor, shape ``(B, data_dim)``
            Observation per chain, in original (un-z-scored) units.
        theta_0 : array/Tensor, shape ``(B, nwalkers, ndim)`` or None
            Walker initialisations in original units. If None, draws from
            the prior.
        mcmc_params : dict, optional
            Same keys as :py:meth:`get_posterior_gpu`.
        return_log_prob : bool
            If True, also return per-sample log-probabilities.

        Returns
        -------
        samples : ndarray, shape ``(B, n_samples, ndim)``
        log_probs : ndarray, shape ``(B, n_samples)`` (only if requested)
        """
        if x_obs.dim() == 1:
            x_obs = x_obs.unsqueeze(0)
        B = x_obs.shape[0]

        if self.z_score:
            x_obs = (x_obs - self.x_mean) / self.x_std

        if self.vmim:
            with torch.inference_mode():
                x_cond = self.embedding_net(x_obs).detach()
        else:
            x_cond = x_obs

        if mcmc_params is not None:
            nwalkers = mcmc_params.get('nwalkers', 1000)
            nsteps = mcmc_params.get('nsteps', 1000)
            discard = mcmc_params.get('discard', 2000)
            thin = mcmc_params.get('thin', 3)
            stretch_a = mcmc_params.get('stretch_a', 2.0)
            progress = mcmc_params.get('progress', True)
        else:
            nwalkers, nsteps, discard, thin, stretch_a, progress = (
                1000, 1000, 2000, 3, 2.0, True
            )

        if nwalkers % 2 != 0:
            raise ValueError(
                "get_posterior_gpu_batched requires an even nwalkers; "
                f"got {nwalkers}."
            )

        tot_steps = nsteps * thin + discard
        ndim = self.parameter_dim
        M = B * nwalkers
        half = nwalkers // 2

        # Buffers allocated OUTSIDE inference_mode so they remain regular
        # tensors; the inference_mode block below writes into them via
        # ``chain[step] = walkers`` without tripping the in-place rule.
        chain_bytes_needed = tot_steps * M * ndim * 4
        existing = getattr(self, '_batched_chain_buf', None)
        if existing is not None and tuple(existing.shape) == (tot_steps, M, ndim):
            store_all = True
        elif existing is not None and tuple(existing.shape) == (nsteps, M, ndim):
            store_all = False
        else:
            if 'cuda' in str(self.device):
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                max_chain_bytes = free_bytes // 2
            else:
                max_chain_bytes = 2**31
            store_all = chain_bytes_needed < max_chain_bytes

        T = tot_steps if store_all else nsteps
        chain = self._get_buf('_batched_chain_buf', (T, M, ndim))
        if return_log_prob:
            lp_chain = self._get_buf('_batched_lp_buf', (T, M))

        step_range = range(tot_steps)
        if progress and trange is not None:
            step_range = trange(tot_steps, desc=f"GPU batched MCMC (B={B})")

        kept = 0
        with torch.inference_mode():
            # zuko-flow log_prob expects input shape
            # (sample_dim, batch_dim, *event_shape); add the leading
            # sample axis so ``input_batch_dim`` matches the condition.
            cond_full = x_cond.repeat_interleave(
                nwalkers, dim=0
            ).unsqueeze(0)                                              # (1, M, cond_dim)

            block_offsets = torch.arange(B, device=self.device) * nwalkers
            in_half = torch.arange(half, device=self.device)
            idx_s0 = (block_offsets[:, None] + in_half[None, :]).flatten()  # (B*half,)
            idx_s1 = idx_s0 + half                                          # (B*half,)
            block_of_active = torch.repeat_interleave(
                torch.arange(B, device=self.device), half
            )
            n_active = B * half

            # Per-half conditioning slices: each active walker keeps its
            # block's x_obs. Precomputed once because idx_s0/idx_s1 are fixed.
            cond_s0 = cond_full[:, idx_s0, :]                          # (1, B*half, cond_dim)
            cond_s1 = cond_full[:, idx_s1, :]

            def log_prob_batch(
                theta: torch.Tensor, cond: torch.Tensor
            ) -> torch.Tensor:
                log_lik = self.density_estimator.log_prob(
                    input=cond, condition=theta
                ).reshape(-1)
                log_prior = self.prior.log_prob(theta)
                return log_lik + log_prior

            if theta_0 is not None:
                walkers = torch.as_tensor(
                    theta_0, device=self.device, dtype=torch.float32
                )
                if walkers.dim() == 3:
                    walkers = walkers.reshape(M, ndim)
                if self.z_score:
                    walkers = (walkers - self.theta_mean) / self.theta_std
            else:
                walkers = self.prior.sample((M,)).to(self.device)

            log_probs = log_prob_batch(walkers, cond_full)             # (M,)

            for step in step_range:
                for active_idx, comp_offset_in_block, cond_active in (
                    (idx_s0, half, cond_s0),
                    (idx_s1, 0, cond_s1),
                ):
                    j = torch.randint(half, (n_active,), device=self.device)
                    partner_idx = (
                        block_of_active * nwalkers + comp_offset_in_block + j
                    )
                    partners = walkers[partner_idx]

                    u = torch.rand(n_active, device=self.device)
                    z = ((stretch_a - 1.0) * u + 1.0) ** 2 / stretch_a

                    proposals = partners + z.unsqueeze(1) * (
                        walkers[active_idx] - partners
                    )

                    lp_prop = log_prob_batch(proposals, cond_active)

                    log_accept = (
                        (ndim - 1) * torch.log(z) + lp_prop - log_probs[active_idx]
                    )
                    accept = torch.log(
                        torch.rand(n_active, device=self.device)
                    ) < log_accept

                    walkers[active_idx[accept]] = proposals[accept]
                    log_probs[active_idx[accept]] = lp_prop[accept]

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

        # Outside inference_mode: views/ops on the regular chain buffer.
        if store_all:
            kept_chain = chain[discard::thin]                       # (nsteps, M, ndim)
            if return_log_prob:
                kept_lp = lp_chain[discard::thin]                   # (nsteps, M)
        else:
            kept_chain = chain[:kept]
            if return_log_prob:
                kept_lp = lp_chain[:kept]

        # (nsteps, B*nwalkers, ndim) -> (B, nsteps*nwalkers, ndim) keeping
        # block ordering. Walkers within a block are contiguous in dim 1.
        nsteps_kept = kept_chain.shape[0]
        kept_chain = kept_chain.reshape(nsteps_kept, B, nwalkers, ndim)
        samples = kept_chain.permute(1, 0, 2, 3).reshape(
            B, nsteps_kept * nwalkers, ndim
        )

        if self.z_score:
            samples = samples * self.theta_std + self.theta_mean

        if return_log_prob:
            kept_lp = kept_lp.reshape(nsteps_kept, B, nwalkers)
            lp_out = kept_lp.permute(1, 0, 2).reshape(
                B, nsteps_kept * nwalkers
            )
            return samples.cpu().numpy(), lp_out.cpu().numpy()

        return samples.cpu().numpy()

    def get_posterior_fiducial(
        self,
        idx: int,
        mcmc_params: Optional[Dict[str, Any]] = None,
        method: str = "emcee",
        return_log_prob: bool = False
    ) -> NDArray:

        x_obs = self.x_fid[idx]

        if mcmc_params is not None:
            nwalkers = mcmc_params.get('nwalkers', 1000)
        else:
            nwalkers = 1000

        # Width set by the parameter-grid std so walkers spread broadly
        # enough to find the posterior while staying well inside the prior.
        # The previous 1% of fiducial was much tighter than typical posterior
        # widths and biased exploration to a small cluster near the fiducial,
        # which can yield artificially narrow posteriors that miss coverage.
        if self.z_score:
            init_scale = 0.3 * self.theta_std.cpu().numpy().flatten()
        else:
            init_scale = 0.3 * self.theta.std(dim=0).cpu().numpy()
        theta_0 = np.random.normal(
            self.theta_fiducial,
            init_scale,
            size=(nwalkers, self.parameter_dim)
        )

        if method == "gpu":
            return self.get_posterior_gpu(
                x_obs, theta_0, mcmc_params, return_log_prob=return_log_prob
            )
        else:
            return self.get_posterior(
                x_obs, theta_0, mcmc_params, return_log_prob=return_log_prob
            )


class HyperparameterTuner:

    def __init__(
        self,
        order: str,
        omitted_bin: Optional[int] = None,
        moped: bool = False,
        params: Optional[List[str]] = None,
        z_score: bool = True,
        device: str = default_device
    ) -> None:

        self.study = None
        self.study_vmim = None
        self.study_nle = None
        self.storage_path = None
        self.model_dir = None
        self.study_name = None
        self.vmim_study_name = None
        self.device = device
        self.order = order
        self.omitted_bin = omitted_bin
        self.moped = moped
        self.vmim = not moped
        self.initialized = False
        self.params = params
        self.z_score = z_score
        self._cached_embedding_net = None
        self._cached_x_grid_embedded = None
        self._cached_x_fid_embedded = None

    def _create_study(
        self,
        study_name: str,
        overwrite: bool,
        load_if_exists: bool,
        sampler: Optional[optuna.samplers.BaseSampler] = None
    ) -> optuna.Study:

        sampler = sampler or optuna.samplers.BaseSampler()
        
        if overwrite:
            try:
                return optuna.create_study(
                    direction="minimize",
                    study_name=study_name,
                    storage=self.storage_path,
                    load_if_exists=False,
                    sampler=sampler
                )
            except optuna.exceptions.DuplicatedStudyError:
                optuna.delete_study(
                    study_name=study_name,
                    storage=self.storage_path
                )
                return optuna.create_study(
                    direction="minimize",
                    study_name=study_name,
                    storage=self.storage_path,
                    load_if_exists=False,
                    sampler=sampler
                )

        return optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=self.storage_path,
            load_if_exists=load_if_exists,
            sampler=sampler
        )
    
    def initialize_study(
        self,
        study_name: str,
        storage_path: str,
        model_dir: str,
        overwrite: bool = False,
        load_if_exists: bool = True,
        sampler_nle: Optional[optuna.samplers.BaseSampler] = None,
        sampler_npe: Optional[optuna.samplers.BaseSampler] = None
    ) -> None:

        self.model_dir = model_dir
        self.study_name = study_name

        self.storage_path = f"sqlite:///{storage_path}"
        
        self.sampler_nle = sampler_nle or optuna.samplers.BaseSampler()
        self.sampler_npe = sampler_npe or optuna.samplers.BaseSampler()

        if self.vmim:
            base_name = study_name[:-1] if study_name.endswith("/") else study_name
            self.vmim_study_name = f"{base_name}_summary"
            self.study_vmim = self._create_study(
                self.vmim_study_name,
                overwrite,
                load_if_exists,
                sampler=self.sampler_npe
            )

        self.study_nle = self._create_study(
            study_name,
            overwrite,
            load_if_exists,
            sampler=self.sampler_nle
        )
        self.study = self.study_nle
        
        self.initialized = True
        
        
    def tune(
        self,
        hyperparam_tuning_dict: Dict[str, Any],
        n_trials: int = 100,
    ) -> None:

        """Tune hyperparameters using Optuna.

        hyperparam_ranges, hyperparam_choices, fixed_params,
        hyperparam_npe_ranges, hyperparam_npe_choices, fixed_params_npe,
        hyperparam_embedding_ranges, hyperparam_embedding_choices, fixed_params_embedding

        """

        assert self.initialized, (
            "Study not initialized. Call 'initialize_study' first."
        )

        def _build_nle_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
            return {
                'model': get_hyperparams('model', 'nle', hyperparam_tuning_dict, trial, default='maf'),
                'num_transforms': get_hyperparams('num_transforms', 'nle', hyperparam_tuning_dict, trial, default=8),
                'num_components': get_hyperparams('num_components', 'nle', hyperparam_tuning_dict, trial, default=8),
                'hidden_features': get_hyperparams('hidden_features', 'nle', hyperparam_tuning_dict, trial, default=256),
                'num_blocks': get_hyperparams('num_blocks', 'nle', hyperparam_tuning_dict, trial, default=2),
                'num_bins': get_hyperparams('num_bins', 'nle', hyperparam_tuning_dict, trial, default=10),
                'learning_rate': get_hyperparams('learning_rate', 'nle', hyperparam_tuning_dict, trial, default=1e-4, log=True, continuous=True),
                'training_batch_size': get_hyperparams('training_batch_size', 'nle', hyperparam_tuning_dict, trial, default=128),
                'stop_after_epochs': get_hyperparams('stop_after_epochs', 'nle', hyperparam_tuning_dict, trial, default=20),
                'patience': get_hyperparams('patience', 'nle', hyperparam_tuning_dict, trial, default=5),
                'factor': get_hyperparams('factor', 'nle', hyperparam_tuning_dict, trial, default=np.sqrt(0.1), continuous=True, log=True),
                'max_num_epochs': get_hyperparams('max_num_epochs', 'nle', hyperparam_tuning_dict, trial, default=1e9),
                'validation_fraction': get_hyperparams('validation_fraction', 'nle', hyperparam_tuning_dict, trial, default=0.1, continuous=True)
            }

        def _build_npe_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
            return {
                'model': get_hyperparams('model', 'npe', hyperparam_tuning_dict, trial, default='mdn'),
                'num_transforms': get_hyperparams('num_transforms', 'npe', hyperparam_tuning_dict, trial, default=6),
                'num_components': get_hyperparams('num_components', 'npe', hyperparam_tuning_dict, trial, default=1),
                'hidden_features': get_hyperparams('hidden_features', 'npe', hyperparam_tuning_dict, trial, default=128),
                'num_blocks': get_hyperparams('num_blocks', 'npe', hyperparam_tuning_dict, trial, default=2),
                'num_bins': get_hyperparams('num_bins', 'npe', hyperparam_tuning_dict, trial, default=10),
                'learning_rate': get_hyperparams('learning_rate', 'npe', hyperparam_tuning_dict, trial, default=1e-4, log=True, continuous=True),
                'training_batch_size': get_hyperparams('training_batch_size', 'npe', hyperparam_tuning_dict, trial, default=128),
                'stop_after_epochs': get_hyperparams('stop_after_epochs', 'npe', hyperparam_tuning_dict, trial, default=20),
                'patience': get_hyperparams('patience', 'npe', hyperparam_tuning_dict, trial, default=5),
                'factor': get_hyperparams('factor', 'npe', hyperparam_tuning_dict, trial, default=np.sqrt(0.1), continuous=True, log=True),
                'max_num_epochs': get_hyperparams('max_num_epochs', 'npe', hyperparam_tuning_dict, trial, default=1e9),
            }

        def _build_embedding_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
            latent_default = len(self.params) * 3 if self.params is not None else 27
            layer_norm = get_hyperparams('layer_norm', 'embedding', hyperparam_tuning_dict, trial, default=True)
            return {
                'hidden_features': get_hyperparams('hidden_features', 'embedding', hyperparam_tuning_dict, trial, default=128),
                'num_blocks': get_hyperparams('num_blocks', 'embedding', hyperparam_tuning_dict, trial, default=2),
                'dropout': get_hyperparams('dropout', 'embedding', hyperparam_tuning_dict, trial, default=0.0),
                'norm_type': 'layer' if layer_norm else None,
                'latent_dim': get_hyperparams('latent_dim', 'embedding', hyperparam_tuning_dict, trial, default=latent_default),
            }

        def objective_vmim(trial: optuna.Trial, max_retries: int = 3) -> float:
            for attempt in range(max_retries):
                hyperparam_dict_npe = _build_npe_hyperparams(trial)
                hyperparam_dict_embedding = _build_embedding_hyperparams(trial)

                tuner_nle = NLE(
                    order=self.order,
                    omitted_bin=self.omitted_bin,
                    moped=self.moped,
                    params=self.params,
                    z_score=self.z_score,
                    device=self.device
                )
                tuner_nle.setup_model(
                    vmim=True,
                    hyperparam_dict=None,
                    hyperparams_embedding=hyperparam_dict_embedding,
                    hyperparams_npe=hyperparam_dict_npe
                )

                ind_dict, _ = get_data_splits_nle_vmim(vmim=True, verbose=False)

                best_vmim_val = tuner_nle.train_vmim_summary(
                    training_params_vmim=hyperparam_dict_npe,
                    use_indices_vmim=ind_dict['use_indices_vmim'],
                    validation_indices_vmim=ind_dict['validation_indices_vmim']
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if math.isfinite(best_vmim_val):
                    trial.set_user_attr('hyperparam_dict_npe', hyperparam_dict_npe)
                    trial.set_user_attr('hyperparam_dict_embedding', hyperparam_dict_embedding)
                    try:
                        current_best = self.study_vmim.best_value
                    except ValueError:
                        current_best = float('inf')
                    if best_vmim_val < current_best:
                        save_model(tuner_nle, self.model_dir / 'best_vmim.pt')
                    return best_vmim_val

                print(f"Trial {trial.number}: got {best_vmim_val}, retrying ({attempt + 1}/{max_retries})")

            raise optuna.TrialPruned(f"Trial {trial.number} returned non-finite loss after {max_retries} retries.")

        def objective_nle(
            trial: optuna.Trial,
            fixed_vmim_hyperparams: Optional[Dict[str, Dict[str, Any]]] = None,
            max_retries: int = 3
        ) -> float:

            for attempt in range(max_retries):
                hyperparam_dict = _build_nle_hyperparams(trial)
                hyperparam_dict_npe = None
                hyperparam_dict_embedding = None

                if self.vmim:
                    if fixed_vmim_hyperparams is None:
                        raise ValueError("VMIM hyperparameters must be provided before NLE tuning.")
                    hyperparam_dict_npe = fixed_vmim_hyperparams['npe']
                    hyperparam_dict_embedding = fixed_vmim_hyperparams['embedding']

                tuner_nle = NLE(
                    order=self.order,
                    omitted_bin=self.omitted_bin,
                    moped=self.moped,
                    params=self.params,
                    z_score=self.z_score,
                    device=self.device
                )
                tuner_nle.setup_model(
                    vmim=self.vmim,
                    hyperparam_dict=hyperparam_dict,
                    hyperparams_embedding=hyperparam_dict_embedding,
                    hyperparams_npe=hyperparam_dict_npe,
                    cached_embedding_net=self._cached_embedding_net,
                    cached_x_grid_embedded=self._cached_x_grid_embedded,
                    cached_x_fid_embedded=self._cached_x_fid_embedded,
                )

                ind_dict, _ = get_data_splits_nle_vmim(vmim=self.vmim, verbose=False)

                best_prob = tuner_nle.train(
                    training_params=hyperparam_dict,
                    training_params_vmim=hyperparam_dict_npe,
                    **ind_dict
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if math.isfinite(best_prob):
                    save_model(tuner_nle, self.model_dir / f"trial_{trial.number}.pt")
                    return best_prob

                print(f"Trial {trial.number}: got {best_prob}, retrying ({attempt + 1}/{max_retries})")

            raise optuna.TrialPruned(f"Trial {trial.number} returned non-finite loss after {max_retries} retries.")

        fixed_vmim_hyperparams = None

        if self.vmim:
            if self.study_vmim is None:
                raise RuntimeError("VMIM study not initialized. Call 'initialize_study' first.")

            vmim_n_trials = hyperparam_tuning_dict.get('vmim_n_trials', n_trials)
            completed_vmim = len(self.study_vmim.trials)
            remaining_vmim = max(0, vmim_n_trials - completed_vmim)

            self.study_vmim.optimize(
                objective_vmim,
                
                n_trials=remaining_vmim,
                n_jobs=1,
                timeout=None,
                gc_after_trial=True
            )

            best_vmim_trial = self.study_vmim.best_trial
            best_npe = best_vmim_trial.user_attrs.get('hyperparam_dict_npe', None)
            best_embedding = best_vmim_trial.user_attrs.get('hyperparam_dict_embedding', None)
            if best_npe is None or best_embedding is None:
                raise RuntimeError("Best VMIM trial does not contain stored hyperparameters.")

            fixed_vmim_hyperparams = {
                'npe': best_npe,
                'embedding': best_embedding
            }

            if (
                self._cached_embedding_net is None
                or self._cached_x_grid_embedded is None
                or self._cached_x_fid_embedded is None
            ):
                best_vmim_path = self.model_dir / 'best_vmim.pt'
                if not best_vmim_path.exists():
                    raise RuntimeError(
                        f"Best VMIM model not found at {best_vmim_path}. "
                        "Re-run the VMIM study so the best trial gets saved, "
                        "or restore the file manually."
                    )
                print(f"Loading saved VMIM model from {best_vmim_path}...")
                cache_holder = load_model(best_vmim_path, device=self.device)
                cache_holder.embedding_net.eval()
                with torch.inference_mode():
                    self._cached_x_grid_embedded = cache_holder.embedding_net(
                        cache_holder.x_grid
                    ).detach()
                    self._cached_x_fid_embedded = cache_holder.embedding_net(
                        cache_holder.x_fid
                    ).detach()
                self._cached_embedding_net = cache_holder.embedding_net
                del cache_holder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        nle_n_trials = hyperparam_tuning_dict.get('nle_n_trials', n_trials)
        completed_nle = len(self.study_nle.trials)
        remaining_nle = max(0, nle_n_trials - completed_nle)

        self.study.optimize(
            lambda trial: objective_nle(trial, fixed_vmim_hyperparams),
            n_trials=remaining_nle,
            n_jobs=1,
            timeout=None,
            gc_after_trial=True
        )
        
        
        