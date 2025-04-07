import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import torch
import argparse
import os
# the sliced package requires numpy version lower than 1.20
from sliced import SlicedInverseRegression

# test methods
from functions_test import fcit_test, cdc_test
from functions_flow import flow_test
from CCIT import CCIT

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--cpu', type=str, default="0-10", help='Indices of cpus for parallel computing.')
    parser.add_argument('--hidden_num', type=int, default=8, help='Hidden dimensions of flow training.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate of flow training.')
    parser.add_argument('--batchsize', type=int, default=200, help='Batchsize of flow training.')
    parser.add_argument('--n_iter', type=int, default=500, help='Iteration of flow training.')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of steps when sampling ODE.')
    parser.add_argument('--FlowCIT_DC', type=int, default=1, help='Implement FlowCIT-DC or not.')
    parser.add_argument('--FlowCIT_IPC', type=int, default=1, help='Implement FlowCIT-IPC or not.')
    parser.add_argument('--FCIT', type=int, default=1, help='Implement FCIT or not.')
    parser.add_argument('--CDC', type=int, default=1, help='Implement CDC or not.')
    parser.add_argument('--CCIT', type=int, default=1, help='Implement CCIT or not.')
    parser.add_argument('--test_size', type=float, default=.1, help='Size to test the conditional independence.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for implementation.')
    # dimension reduction methods
    parser.add_argument('--method', type=str, default="pca", help='Method to do dimension reduction.')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension required.')
    return parser.parse_args()

def load_data(file_path, target_column, standardize=True):
    df = pd.read_csv(file_path, delimiter=";")
    y = df[[target_column]]
    X = df.drop(columns=[target_column])
    if standardize:
        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)
        y_scaler = scaler.fit_transform(y)
        return X_scaler, y_scaler
    else:
        return X, y


def save_to_local(x, y, z, postfix):  
    # Convert to DataFrames with column labels
    df_x = pd.DataFrame(x.numpy(), columns=[f"x_{i}" for i in range(x.shape[1])])
    df_y = pd.DataFrame(y.numpy(), columns=[f"y_{i}" for i in range(y.shape[1])])
    df_z = pd.DataFrame(z.numpy(), columns=[f"z_{i}" for i in range(z.shape[1])])
    # Save to CSV
    df_x.to_csv(f"/home/chenxhe/flow_test/data/x_{postfix}.csv", index=False)
    df_y.to_csv(f"/home/chenxhe/flow_test/data/y_{postfix}.csv", index=False)
    df_z.to_csv(f"/home/chenxhe/flow_test/data/z_{postfix}.csv", index=False)


def test(x, y, z, args, device):
    seed = args.seed
    device = device

    batchsize = args.batchsize
    n_iter = args.n_iter
    hidden_num = args.hidden_num
    lr = args.lr
    num_steps = args.num_steps
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if args.CDC:
        _, p_cdc =  cdc_test(x, y, z)
        print("CDC test, p_value:", p_cdc)
    if args.FCIT:
        _, p_fcit = fcit_test(x, y, z)
        print("FCIT test, p_value:", p_fcit)
    if args.CCIT:
        p_ccit = CCIT.CCIT(x.numpy(), y.numpy(), z.numpy())
        print("CCIT test, p_value:", p_ccit)
    if args.FlowCIT_DC:
        _, p_flowcit_dc = flow_test(x=x, y=y, z=z, permutation=1, method="DC", seed=seed, batchsize=batchsize, n_iter=n_iter, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
        print("FlowCIT-DC test, p_value:", p_flowcit_dc)
    if args.FlowCIT_IPC:
        _, p_flowcit_ipc = flow_test(x=x, y=y, z=z, permutation=1, method="IPC", seed=seed, batchsize=batchsize, n_iter=n_iter,  hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
        print("FlowCIT-IPC test, p_value:", p_flowcit_ipc)


if __name__=="__main__":
    args = parse_arguments()
    # parameters to be set
    seed = args.seed
    test_size = args.test_size

    if args.gpu:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        device = "cpu"
    
    X, y = load_data(file_path="/home/chenxhe/flow_test/real_data_wine/winequality-white.csv", target_column="quality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    method = args.method
    latent_dim = args.latent_dim
    if args.method == "full":
        postfix = "full"
    else:
        postfix = f"{method}_{latent_dim}"

    X_test_torch = torch.from_numpy(X_test).float()
    y_test_torch = torch.from_numpy(y_test).float()

    # baseline: X \indep Y \mid X, which is definitely H_0
    if method == "full":
        X_test_dr = X_test_torch
    elif method == "pca":
        pca = PCA(n_components=latent_dim).fit(X_train)
        X_test_dr = torch.from_numpy(pca.transform(X_test)).float()
    elif method == "sir":
        sir = SlicedInverseRegression(n_directions=latent_dim).fit(X_train, y_train.squeeze())
        X_test_dr = torch.from_numpy(sir.transform(X_test)).float()
    elif method == "umap":
        umap_model = umap.UMAP(n_neighbors=100, n_components=latent_dim, min_dist=0.5, metric="euclidean").fit(X_train)
        X_test_dr = torch.from_numpy(umap_model.transform(X_test))
    elif method == "ddr":
        X_test_dr = torch.from_numpy(pd.read_csv(f'results_ddr/test_results_epoch_z_dim{latent_dim}_ep25.csv').to_numpy()).float()

    save_to_local(X_test_torch, y_test_torch, X_test_dr, postfix=postfix)
    test(x=X_test_torch, y=y_test_torch, z=X_test_dr, args=args, device=device)
