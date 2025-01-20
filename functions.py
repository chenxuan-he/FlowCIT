import torch
import numpy as np
import scipy.stats as stats
from hyppo.conditional import FCIT, KCI, PartialDcorr, ConditionalDcorr
from hyppo.tools import linear, correlated_normal
from sklearn.tree import DecisionTreeRegressor


def FCIT(x, y, z, seed):
    """Fast conditional independence test."""
    np.random.seed(seed)
    dim = 2
    n = 100000
    z1 = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n))
    A1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
    B1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
    x1 = (A1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
    y1 = (B1 @ z1.T + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=(n)).T)
    model = DecisionTreeRegressor()
    cv_grid = {"min_samples_split": [2, 8, 64, 512, 1e-2, 0.2, 0.4]}
    stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x1.T, y1.T, z1)
    stat, pvalue = FCIT(model=model, cv_grid=cv_grid).test(x.T, y.T, z)
    return stat, pvalue


def KCI(x, y, z, seed):
    """Kernel-based conditional independence test."""
    np.random.seed(seed)
    stat, pvalue = KCI().test(x, y, z)
    return stat, pvalue


def PDC(x, y, z, seed):
    """Partial distance correlation."""
    np.random.seed(seed)
    stat, pvalue = PartialDcorr().test(x, y, z)
    return stat, pvalue


def CDC(x, y, z, seed):
    """Conditional distance correlation."""
    np.random.seed(seed)
    stat, pvalue = ConditionalDcorr().test(x, y, z)
    return stat, pvalue


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

