# This is the code to perform various test for conditional independence test.
import torch
import numpy as np
import scipy.stats as stats
from hyppo.conditional import FCIT, PartialDcorr, ConditionalDcorr
from sklearn.tree import DecisionTreeRegressor

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
