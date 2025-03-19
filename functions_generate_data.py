import numpy as np
import pandas as pd
import torch
import random
from sklearn.datasets import make_swiss_roll

def generate_data(model=1, sim_type=0, n=1000, p=3, q=3, d=3, alpha=.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if model==1:
        Z = torch.randn((n, d))
        Z1 = Z[:,0].unsqueeze(1)
        Z2 = Z[:,1].unsqueeze(1)
        tilde_X = torch.randn((n, 1))
        if sim_type==1:
            X = tilde_X + Z1 + Z2
            Y = alpha * tilde_X + Z1
        elif sim_type==2:
            X = torch.log(tilde_X * Z1 + 10) + Z2
            Y = alpha * torch.exp(tilde_X * Z1) + Z2
    # # model 1: dense matrix
    # if model==1:
    #     beta_1 = torch.randn((d, p))
    #     beta_2 = torch.randn((d, q))
    #     beta_3 = torch.randn((p, q))
    # # model 2: sparse matrix
    # elif model==2:
    #     beta_1 = torch.zeros((d, p))
    #     beta_2 = torch.zeros((d, q))
    #     beta_3 = torch.zeros((p, q))
    #     # Set the first 3x3 block to random values
    #     beta_1[0:3, 0:3] = torch.randn((3, 3))
    #     beta_2[0:3, 0:3] = torch.randn((3, 3))
    #     beta_3[0:3, 0:3] = torch.randn((3, 3))
    # elif model==3:
    #     if sim_type==1:
    #         Z = generate_swiss_roll(n, dim=d, seed=seed)
    #     elif sim_type==2:
    #         Z = generate_helix(n, dim=d, seed=seed)
    #     epsilon_X = np.random.normal(0, 1, Z.shape)
    #     epsilon_Y = np.random.normal(0, 1, Z.shape)
    #     X = Z + epsilon_X
    #     # Independent case
    #     if alpha == 0:
    #         Y = Z + epsilon_Y
    #     # Dependent case
    #     else:
    #         Y = alpha * X + (1 - alpha) * Z + epsilon_Y
    #     return torch.from_numpy(np.array(X, dtype=np.float32)), torch.from_numpy(np.array(Y, dtype=np.float32)), torch.from_numpy(np.array(Z, dtype=np.float32))
    # else:
    #     return 0
    # # Generate Z and X
    # Z = torch.randn((n, d))
    # if sim_type == 1:
    #     X = Z @ beta_1 + torch.randn((n, p))
    #     Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    # elif model == 1 and sim_type == 2:
    #     X = Z @ beta_1 + torch.randn((n, p))
    #     Y = torch.sin(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    # elif model == 2 and sim_type == 2:
    #     X = torch.abs(Z @ beta_1) + torch.randn((n, p))
    #     Y = (Z @ beta_2) + X @ beta_3 * alpha + torch.randn((n, q))
    # elif sim_type == 3:
    #     X = (Z @ beta_1) + torch.randn((n, p))
    #     # Y = torch.cos(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    #     Y = torch.pow(torch.abs(Z @ beta_2), 3/2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    # elif sim_type == 4:
    #     X = Z @ beta_1 + torch.randn((n, p))
    #     Y = Z @ beta_2 + torch.exp(X @ beta_3 * alpha) + torch.randn((n, q))

    df = pd.DataFrame(
        torch.cat([X, Y, Z], dim=1).numpy(),
        columns=[f"X{i+1}" for i in range(p)] + [f"Y{i+1}" for i in range(q)] + [f"Z{i+1}" for i in range(d)]
    )
    df.to_csv(f"data/data_model{model}_simtype{sim_type}_alpha{alpha}_n{n}_p{p}_q{q}_d{d}_seed{seed}.csv", index=False)
    return X, Y, Z


def read_data(model, sim_type, alpha, n, p, q, d, seed):
    # Load CSV
    df = pd.read_csv(f"data/data_model{model}_simtype{sim_type}_alpha{alpha}_n{n}_p{p}_q{q}_d{d}_seed{seed}.csv")
    # Convert back to PyTorch tensors
    X = torch.tensor(df[[f"X{i+1}" for i in range(p)]].values)
    Y = torch.tensor(df[[f"Y{i+1}" for i in range(q)]].values)
    Z = torch.tensor(df[[f"Z{i+1}" for i in range(d)]].values)
    return X, Y, Z


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
