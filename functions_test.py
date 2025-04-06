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


# Code to compute improved projection correlation (python)
# Y. Zhang and L. Zhu, “Projective independence tests in high dimensions: the curses and the cures,” Biometrika, vol. 111, no. 3, pp. 1013–1027, Sep. 2024, doi: 10.1093/biomet/asad070.
# ustat: converges to standard normal under null; IPC2: squared IPC between x and y 
def IPC(x, y, alpha=.05):
    n = y.shape[0]

    xxup = x @ x.T + np.sum(np.mean(x**2, axis=0))
    xxdown = np.sqrt(np.diag(xxup))
    xxdown = np.outer(xxdown, xxdown)
    b0 = np.arcsin(xxup / xxdown + 0j)
    b = np.real(b0)
    np.fill_diagonal(b, 0)
    bjsum = np.sum(b, axis=1)
    bbsum = np.sum(bjsum)

    yyup = y @ y.T + np.sum(np.mean(y**2, axis=0))
    yydown = np.sqrt(np.diag(yyup))
    yydown = np.outer(yydown, yydown)
    rhok0 = np.arcsin(yyup / yydown + 0j)
    rhok = np.real(rhok0)
    np.fill_diagonal(rhok, 0)
    rhoksum = np.sum(rhok, axis=1)
    rhosum = np.sum(rhoksum)

    S1 = np.mean(np.sum(b * rhok, axis=1)) / (n - 3)
    S2 = np.mean(bjsum * rhoksum) / ((n - 3) * (n - 2))
    S3 = rhosum * bbsum / (n * (n - 3) * (n - 2) * (n - 1))
    CVM2 = S1 - 2 * S2 + S3

    S1 = np.mean(np.sum(b * b, axis=1)) / (n - 3)
    S2 = np.mean(bjsum * bjsum) / ((n - 3) * (n - 2))
    S3 = bbsum * bbsum / (n * (n - 3) * (n - 2) * (n - 1))
    vx = S1 - 2 * S2 + S3

    S1 = np.mean(np.sum(rhok * rhok, axis=1)) / (n - 3)
    S2 = np.mean(rhoksum * rhoksum) / ((n - 3) * (n - 2))
    S3 = rhosum * rhosum / (n * (n - 3) * (n - 2) * (n - 1))
    vy = S1 - 2 * S2 + S3

    IPC2 = CVM2 / np.sqrt(vx * vy)
    ustat = n * CVM2 / np.sqrt(2 * vx * vy)

    rej = (np.abs(ustat) > (stats.norm.ppf(1-alpha/2)))
    return ustat, IPC2, rej


def permutation_test(X, Y, num_permutations=100, seed=0, permutation=1, method="DC"):
    """Perform a permutation test to assess the significance of the distance correlation."""
    torch.manual_seed(seed)
    if method=="DC":
        observed_dcor, _, rej = distance_correlation(X, Y)
        # print(observed_dcor)
        if not permutation:
            return observed_dcor.item(), 1-rej
        n = X.size(0)
        
        permuted_dcors = []
        for i in range(num_permutations):
            permuted_Y = Y[torch.randperm(n)]                
            permuted_dcor, _, _ = distance_correlation(X, permuted_Y)
            # if i == 0 or i == 1:
            #     print(permuted_dcor)
            permuted_dcors.append(permuted_dcor.item())
        
        permuted_dcors = torch.tensor(permuted_dcors)
        p_value = (permuted_dcors >= observed_dcor).float().mean().item()
        
        return observed_dcor.item(), p_value
    elif method=="IPC":
        observed_dcor, _, rej = IPC(X.numpy(), Y.numpy())
        if not permutation:
            return observed_dcor.item(), 1-rej
        n = X.size(0)
        
        permuted_dcors = []
        for _ in range(num_permutations):
            permuted_Y = Y[torch.randperm(n)]
            permuted_dcor, _, _ = IPC(X.numpy(), permuted_Y.numpy())
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
