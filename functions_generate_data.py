import numpy as np
import pandas as pd
import torch
import random
import os
from sklearn.datasets import make_swiss_roll

def generate_data(model=1, sim_type=0, n=1000, p=3, q=3, d=3, s=2, alpha=.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # model 1: dense matrix
    if model==1 or model==3:
        beta_1 = torch.randn((d, p))
        beta_2 = torch.randn((d, q))
        beta_3 = torch.randn((p, q))
    # model 2: sparse matrix
    elif model==2:
        beta_1 = torch.zeros((d, p))
        beta_2 = torch.zeros((d, q))
        beta_3 = torch.zeros((p, q))
        # Set the first 3x3 block to random values
        if sim_type==2:
            beta_1[0:s, 0:s] = torch.randn((s, s))
            beta_2[0:s, 0:s] = torch.randn((s, s))
            beta_3[0:s, 0:s] = torch.randn((s, s))
        elif sim_type==1:
            beta_1[0:s, 0:p] = torch.randn((s, p))
            beta_2[0:s, 0:q] = torch.randn((s, q))
            beta_3[0:p, 0:q] = torch.randn((p, q))
        elif sim_type==4:
            # beta_1 = torch.randn((d, p))
            # beta_2 = torch.randn((d, q))
            # beta_3 = torch.randn((p, q))
            beta_1 = torch.where(torch.bernoulli(torch.full((d, p), 0.1)).to(torch.bool), .1*torch.randn((d, p)), torch.zeros_like(torch.randn((d, p))))
            beta_2 = torch.where(torch.bernoulli(torch.full((d, q), 0.1)).to(torch.bool), .1*torch.randn((d, q)), torch.zeros_like(torch.randn((d, q))))
            beta_3 = torch.where(torch.bernoulli(torch.full((p, q), 0.1)).to(torch.bool), .1*torch.randn((p, q)), torch.zeros_like(torch.randn((p, q))))

    # model 4: only Z is high-dimensional and sparse
    elif model==4: 
        beta_1 = torch.zeros((d, p))
        beta_2 = torch.zeros((d, q))
        beta_3 = torch.randn((p, q))
        # Set the first 3 elements of Z is influencing X and Y
        beta_1[0:s, 0:p] = torch.randn((s, p))
        beta_2[0:s, 0:q] = torch.randn((s, q))
    # Now start generate X, Y, and Z.
    Z = torch.randn((n, d))
    if sim_type == 1 or (model==4 and sim_type==3):
        X = Z @ beta_1 + torch.randn((n, p))
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif model == 1 and sim_type == 2:
        X = Z @ beta_1 + torch.randn((n, p))
        Y = torch.sin(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    elif model == 1 and sim_type == 3:
        X = torch.pow(Z @ beta_1, 2) + torch.randn((n, p))
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif (model == 1 and sim_type == 4) or (model == 3 and sim_type == 2):
        X = Z @ beta_1 + torch.randn((n, p))
        Y = Z @ beta_2 + torch.exp(X @ beta_3 * alpha) + torch.randn((n, q))
    elif model == 2 and sim_type == 2:
        X = torch.abs(Z @ beta_1) + torch.randn((n, p))
        Y = (Z @ beta_2) + X @ beta_3 * alpha + torch.randn((n, q))
    elif model == 2 and sim_type == 3:
        X = (Z @ beta_1) + torch.randn((n, p))
        Y = torch.cos(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
        # Y = torch.pow(torch.abs(Z @ beta_2), 3/2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    elif model == 2 and sim_type == 4:
        X = torch.cos(Z @ beta_1) + torch.randn((n, p))
        Y = (Z @ beta_2) + torch.sin(X @ beta_3 * alpha) + torch.randn((n, q))
    elif model == 3 and sim_type == 3:
        t_dist = torch.distributions.StudentT(3)
        X = Z @ beta_1 + t_dist.sample((n, p))
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif model == 3 and sim_type == 4:
        t_dist = torch.distributions.StudentT(3)
        X = torch.cos(Z @ beta_1) + t_dist.sample((n, p))
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif model == 4 and sim_type == 2:
        X = Z @ beta_1 + torch.randn((n, p))
        Y = torch.pow(Z @ beta_2, 2)+ X @ beta_3 * alpha + torch.randn((n, q))
    df = pd.DataFrame(
        torch.cat([X, Y, Z], dim=1).numpy(),
        columns=[f"X{i+1}" for i in range(p)] + [f"Y{i+1}" for i in range(q)] + [f"Z{i+1}" for i in range(d)]
    )
    df.to_csv(f"data/data_model{model}_simtype{sim_type}_alpha{alpha}_n{n}_p{p}_q{q}_d{d}_seed{seed}.csv", index=False)
    return X, Y, Z


def read_data(model, sim_type, alpha, n, p, q, d, seed):
    df = pd.read_csv(f"/home/chenxhe/flow_test/data/data_model{model}_simtype{sim_type}_alpha{alpha}_n{n}_p{p}_q{q}_d{d}_seed{seed}.csv")
    # Convert back to PyTorch tensors
    X = torch.tensor(df[[f"X{i+1}" for i in range(p)]].values)
    Y = torch.tensor(df[[f"Y{i+1}" for i in range(q)]].values)
    Z = torch.tensor(df[[f"Z{i+1}" for i in range(d)]].values)
    return X.type(torch.float32), Y.type(torch.float32), Z.type(torch.float32)


def generate_swiss_roll(n_samples, dim=3, noise=0.05, seed=0):
    """
    Generate data in the shape of a Swiss Roll.
    """
    np.random.seed(seed)
    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    if dim > 3:
        extra_dims = np.random.randn(n_samples, dim - 3)
        return np.hstack((data, extra_dims))
    return data


def generate_helix(n_samples, dim=3, noise=0.1, seed=0):
    """
    Generate data in the shape of a 3D helix.
    """
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    data = np.stack((x, y, z), axis=1)
    if dim > 3:
        extra_dims = np.random.randn(n_samples, dim - 3) * noise
        return np.hstack((data, extra_dims))
    return data + np.random.normal(scale=noise, size=data.shape)
