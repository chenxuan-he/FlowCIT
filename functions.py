import torch
import numpy as np
import scipy.stats as stats
from hyppo.conditional import FCIT, KCI, PartialDcorr, ConditionalDcorr
from sklearn.tree import DecisionTreeRegressor
import random

def generate_data(sim_type=0, n=1000, p=3, q=3, d=3, alpha=.1, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # Under H0: X is independent of Y given Z
    beta_1 = torch.randn((d, p))
    beta_2 = torch.randn((d, q))
    beta_3 = torch.randn((p, q))
    # Generate Z and X
    Z = torch.randn((n, d))
    X = Z @ beta_1 + torch.randn((n, p))
    if (sim_type == 0):
        # Type 0: Under H_0, generate X and Y independently given Z
        Y = Z @ beta_2 + torch.randn((n, q))
    elif (sim_type == 1):
        # Type 1: Under H1, generate X is not independent of Y given Z
        Y = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    elif (sim_type == 2):
        # Type 2: Under H1, generate X is not independent of Y given Z with nonlinear relationship
        Y = torch.sin(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
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


def permutation_test(X, Y, num_permutations=1000, seed=0):
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


def pdc_test(x, y, z, seed=0):
    """Partial distance correlation."""
    np.random.seed(seed)
    x0 = x.cpu().numpy()
    y0 = y.cpu().numpy()
    z0 = z.cpu().numpy()
    stat, pvalue = PartialDcorr().test(x0, y0, z0)
    return stat, pvalue


def cdc_test(x, y, z, seed=0):
    """Conditional distance correlation."""
    x0 = x.cpu().numpy()
    y0 = y.cpu().numpy()
    z0 = z.cpu().numpy()
    np.random.seed(seed)
    stat, pvalue = ConditionalDcorr().test(x0, y0, z0)
    return stat, pvalue
