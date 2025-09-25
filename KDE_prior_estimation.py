import torch
from torch.distributions import Distribution
import math
import numpy as np

class KDEDistribution(Distribution):
    def __init__(self, samples, bandwidth=0.1, device='cpu'):
        self.device = device
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(samples, dtype=torch.float32, device=device)
        else:
            samples = samples.to(device)
        self.raw_samples = samples  # Unstandardized original samples
        # Compute standardization parameters
        self.sample_mean = samples.mean(0)
        self.sample_std = samples.std(0)
        # Apply standardization
        self.samples = self.standardize(samples)  # shape: [N, D]
        self.N, self.D = self.samples.shape
        self.bandwidth = bandwidth
        super().__init__(validate_args=False)
    
    def standardize(self, x):
        return (x - self.sample_mean) / self.sample_std
    
    def destandardize(self, x):
        return x * self.sample_std + self.sample_mean
    
    def sample(self, sample_shape=torch.Size()):
        if isinstance(sample_shape, tuple):
            n_samples = int(torch.tensor(sample_shape).prod().item())
        else:
            n_samples = int(sample_shape)
        idx = torch.randint(0, self.N, (n_samples,), device=self.device)
        noise = torch.randn(n_samples, self.D, device=self.device) * self.bandwidth
        standardized_samples = self.samples[idx] + noise
        return self.destandardize(standardized_samples)
    
    def log_prob(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)
        x_standardized = self.standardize(x)
        x_exp = x_standardized.unsqueeze(-2)          # [..., 1, D]
        s_exp = self.samples.unsqueeze(0)             # [1, N, D]
        dist_sq = ((x_exp - s_exp) ** 2).sum(-1)      # [..., N]
        kernel = torch.exp(-0.5 * dist_sq / self.bandwidth**2)
        norm = (2 * math.pi * self.bandwidth ** 2) ** (self.D / 2)
        return torch.log(kernel.mean(-1) / norm)