import numpy as np

def IPC(x, y):
    n = y.shape[0]

    xxup = x @ x.T + np.sum(np.mean(x**2, axis=0))
    xxdown = np.sqrt(np.diag(xxup))
    xxdown = np.outer(xxdown, xxdown)
    b0 = np.arcsin(xxup / xxdown)
    b = np.real(b0)
    np.fill_diagonal(b, 0)
    bjsum = np.sum(b, axis=1)
    bbsum = np.sum(bjsum)

    yyup = y @ y.T + np.sum(np.mean(y**2, axis=0))
    yydown = np.sqrt(np.diag(yyup))
    yydown = np.outer(yydown, yydown)
    rhok0 = np.arcsin(yyup / yydown)
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

    return ustat, IPC2
