import torch
import scipy.stats as stats

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
    return dcorr, test_stat, rej  # Avoid division by zero

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

