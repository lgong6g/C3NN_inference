import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex":True,
    "font.family":"Helvetica",
    "font.size"  : 13,
})

# import equivariant steerable cnn
import escnn.gspaces
import escnn.nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from sklearn.model_selection import train_test_split

from tqdm import tqdm
# Checking if you have a GPU and CUDA set up -- if not, training will take longer by running on the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
# device = 'cpu'
# Set the plotting style
#plt.style.use('seaborn-poster')

from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn

from os import listdir
from os.path import isfile, join

from copy import deepcopy

#from getdist import MCSamples, plots

import pickle as pk

from hyperparameters import hparams, hparams_cnn

import sys
sys.path.append('./src')
from hyperparameters import hparams
from C3NN import C3NN
from load_data import load_data_wide
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
import numpy as np
import random
from torch import from_numpy
import torch
import time, datetime
import cloudpickle as cpk

training_parameters, training_data = load_data_wide(hparams)

# CNN we've been using thus far
from hyperparameters import hparams_cnn
from CNN import CNNEmbedding
hparams_cnn.bottleneck=3
model = CNNEmbedding(hparams_cnn)

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import math

def train_model(model, train_loader, val_loader, device,
                num_epochs=100,
                criterion=None,
                optimizer=None,
                lr=1e-3,
                # Early stopping:
                es_patience=10,
                es_min_delta=0.0,
                # LR scheduler on plateau:
                lr_factor=0.1,
                lr_patience=5,
                lr_min=1e-6,
                lr_threshold=1e-4):
    """
    Trains a regression model with MSE loss and reports MSE, RMSE, and MAE.
    Adds: Early stopping (on val MSE) and ReduceLROnPlateau scheduler.

    Returns:
        model (best weights restored), history (list of dicts per epoch)
    """
    model = model.to(device)
    if criterion is None:
        criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        threshold_mode='rel',
        min_lr=lr_min,
        verbose=False
    )

    print('Starting Training...')
    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    history = []

    for epoch in range(num_epochs):
        # -------- TRAIN --------
        model.train()
        train_mse_sum, train_mae_sum, n_train = 0.0, 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                bsz = y.size(0)
                n_train += bsz
                train_mse_sum += torch.mean((pred - y) ** 2).item() * bsz
                train_mae_sum += torch.mean(torch.abs(pred - y)).item() * bsz

        train_mse = train_mse_sum / n_train
        train_rmse = train_mse ** 0.5
        train_mae = train_mae_sum / n_train

        # -------- VALIDATION --------
        model.eval()
        val_mse_sum, val_mae_sum, n_val = 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_mse_sum += torch.mean((pred - y) ** 2).item() * y.size(0)
                val_mae_sum += torch.mean(torch.abs(pred - y)).item() * y.size(0)
                n_val += y.size(0)

        val_mse = val_mse_sum / n_val
        val_rmse = val_mse ** 0.5
        val_mae = val_mae_sum / n_val

        # ---- Scheduler step on validation metric ----
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']

        # ---- Logging ----
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train: MSE {train_mse:.6f} | RMSE {train_rmse:.6f} | MAE {train_mae:.6f} || "
            f"Val: MSE {val_mse:.6f} | RMSE {val_rmse:.6f} | MAE {val_mae:.6f} || "
            f"LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_mse": train_mse, "train_rmse": train_rmse, "train_mae": train_mae,
            "val_mse": val_mse, "val_rmse": val_rmse, "val_mae": val_mae,
            "lr": current_lr
        })

        # ---- Early stopping (on val_mse) ----
        if val_mse < (best_val - es_min_delta):
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= es_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val MSE: {best_val:.6f}")
                break

    # restore best weights
    model.load_state_dict(best_state)
    return model, history


training_parameters_mean = training_parameters.mean(0)
training_parameters_std = training_parameters.std(0)
training_parameters_stand = (training_parameters - training_parameters_mean) / training_parameters_std

X_train, X_val, y_train, y_val = train_test_split(training_data, training_parameters_stand, test_size=0.1, random_state=42)
#sys.exit()
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
device = 'cuda'
model, history = train_model(model, train_loader, val_loader, device, num_epochs=200, lr=1e-2, es_patience=20)


import cloudpickle as cpk
with open(f"/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/imported_model_1k_examples.cpk", "wb") as handle:
    cpk.dump(model, handle)



