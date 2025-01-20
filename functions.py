import torch
import numpy as np
import scipy.stats as stats
from hyppo.conditional import FCIT, KCI, PartialDcorr, ConditionalDcorr
from hyppo.tools import linear, correlated_normal
from sklearn.tree import DecisionTreeRegressor
import random


def generate_data(n=1000, p=200, q=200, d=200, alpha=.1, seed=0):
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
    # Generate X and Y independently given Z
    Y_H0 = Z @ beta_2 + torch.randn((n, q))
    # Under H1: X is not independent of Y given Z
    Y_H1 = Z @ beta_2 + X @ beta_3 * alpha + torch.randn((n, q))
    Y_H1_nonlinear = torch.sin(Z @ beta_2) + (X @ beta_3 * alpha) + torch.randn((n, q))
    return (X, Y_H0, Z), (X, Y_H1, Z), (X, Y_H1_nonlinear, Z)


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


def FCIT(x, y, z, seed=0):
    """Fast conditional independence test."""
    np.random.seed(seed)
    model = DecisionTreeRegressor()
    cv_grid = {"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]}
    stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x.T, y.T, z)
    return stat, pvalue


def KCI(x, y, z, seed=0):
    """Kernel-based conditional independence test."""
    np.random.seed(seed)
    stat, pvalue = KCI().test(x, y, z)
    return stat, pvalue


def PDC(x, y, z, seed=0):
    """Partial distance correlation."""
    np.random.seed(seed)
    stat, pvalue = PartialDcorr().test(x, y, z)
    return stat, pvalue


def CDC(x, y, z, seed=0):
    """Conditional distance correlation."""
    np.random.seed(seed)
    stat, pvalue = ConditionalDcorr().test(x, y, z)
    return stat, pvalue
