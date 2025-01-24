# This is the code to perform various test for conditional independence test.
import torch
import numpy as np
import scipy.stats as stats
from hyppo.conditional import FCIT, PartialDcorr, ConditionalDcorr
from sklearn.tree import DecisionTreeRegressor
import random

def generate_data(model=1, sim_type=0, n=1000, p=3, q=3, d=3, alpha=.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # model 1: dense matrix
    if model==1:
        beta_1 = torch.randn((d, p))
        beta_2 = torch.randn((d, q))
        beta_3 = torch.randn((p, q))
    # model 2: sparse matrix
    elif model==2:
        beta_1 = torch.zeros((d, p))
        beta_2 = torch.zeros((d, q))
        beta_3 = torch.zeros((p, q))
        # Set the first 3x3 block to random values
        beta_1[0:3, 0:3] = torch.randn((3, 3))
        beta_2[0:3, 0:3] = torch.randn((3, 3))
        beta_3[0:3, 0:3] = torch.randn((3, 3))
    else:
        return 0
    # Generate Z and X
    Z = torch.randn((n, d))
    if sim_type == 1:
        X = Z @ beta_1 + torch.randn((n, p))
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif sim_type == 2:
        X = Z @ beta_1 + torch.randn((n, p))
        Y = torch.sin(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    elif sim_type == 3:
        X = torch.sin(Z @ beta_1) + torch.randn((n, p))
        Y = (Z @ beta_2) + torch.power(X @ beta_3 * alpha, 2) + torch.randn((n, q))
    elif sim_type == 4:
        X = torch.power(Z @ beta_1, 2) + torch.randn((n, p))
        Y = torch.power(Z @ beta_2, 2) + torch.sin(X @ beta_3 * alpha) + torch.randn((n, q))
    return X, Y, Z


def distance_correlation(eps1, eps2, alpha=.05):
    """Compute the distance correlation between two sets of samples."""
    n = eps1.shape[0]
    a = torch.cdist(eps1, eps1, p=2)
    b = torch.cdist(eps2, eps2, p=2)
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    dcov = torch.sqrt((A * B).mean())
    dvar_x = torch.sqrt((A * A).mean())
    dvar_y = torch.sqrt((B * B).mean())
    dcorr = dcov / torch.sqrt(dvar_x * dvar_y + 1e-10)
    test_stat = n*dcorr**2
    rej = (test_stat > (stats.norm.ppf(1-alpha/2))**2)
    return dcorr, test_stat, rej 


def permutation_test(X, Y, num_permutations=100, seed=0):
    """Perform a permutation test to assess the significance of the distance correlation."""
    torch.manual_seed(seed)
    observed_dcor, _, _ = distance_correlation(X, Y)
    n = X.size(0)
    
    permuted_dcors = []
    for _ in range(num_permutations):
        permuted_Y = Y[torch.randperm(n)]
        permuted_dcor, _, _ = distance_correlation(X, permuted_Y)
        permuted_dcors.append(permuted_dcor.item())
    
    permuted_dcors = torch.tensor(permuted_dcors)
    p_value = (permuted_dcors >= observed_dcor).float().mean().item()
    
    return observed_dcor.item(), p_value


def fcit_test(x, y, z, seed=0):
    """Fast conditional independence test."""
    np.random.seed(seed)
    model = DecisionTreeRegressor()
    cv_grid = {"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]}
    stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x, y, z)
    return stat, pvalue


def pdc_test(x, y, z, seed=0, reps=100, workers=-1):
    """Partial distance correlation."""
    np.random.seed(seed)
    x0 = x.cpu().numpy()
    y0 = y.cpu().numpy()
    z0 = z.cpu().numpy()
    stat, pvalue = PartialDcorr().test(x0, y0, z0, reps=reps, workers=workers, random_state=seed)
    return stat, pvalue


def cdc_test(x, y, z, seed=0, reps=100, workers=-1):
    """Conditional distance correlation."""
    x0 = x.cpu().numpy()
    y0 = y.cpu().numpy()
    z0 = z.cpu().numpy()
    np.random.seed(seed)
    stat, pvalue = ConditionalDcorr().test(x0, y0, z0, reps=reps, workers=workers, random_state=seed)
    return stat, pvalue
