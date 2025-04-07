import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
import torch
import argparse
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# local functions
from DDR.model_reg import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--cpu', type=str, default="0-10", help='Indices of cpus for parallel computing.')
    parser.add_argument('--hidden_num', type=int, default=8, help='Hidden dimensions of flow training.')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate of flow training.')
    parser.add_argument('--batchsize', type=int, default=200, help='Batchsize of flow training.')
    parser.add_argument('--n_iter', type=int, default=100, help='Iteration of flow training.')
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
    parser.add_argument('--save', type=str, default="results_ddr", help='Latent dimension required.')
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


if __name__=="__main__":
    args = parse_arguments()
    # parameters to be set
    seed = args.seed
    test_size = args.test_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if args.gpu:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        device = "cpu"
    
    X, y = load_data(file_path="/home/chenxhe/flow_test/real_data_wine/winequality-white.csv", target_column="quality")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # 4. deep dimension reduction
    train_dat = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    trainLoader = DataLoader(train_dat, batch_size=args.batchsize, shuffle=True)
    test_dat = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    testLoader = DataLoader(test_dat, batch_size=args.batchsize, shuffle=False)
    testLoader_cor = DataLoader(test_dat, batch_size=len(test_dat), shuffle=False)
    D_net = Discriminator(ndim = args.latent_dim)
    net = Generator(xdim = X.shape[1], ndim = args.latent_dim)
    
    print('  + Number of params (net) : {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    print('  + Number of params (Dnet) : {}'.format(
        sum([p.data.nelement() for p in D_net.parameters()])))
    if device != "cpu":
        net = net.cuda()
        D_net = D_net.cuda()
        
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    optimizer_D = optim.Adam(D_net.parameters(), weight_decay=1e-4)

    for epoch in range(1, args.n_iter + 1):
        if epoch < 150: zlr = 3.0
        elif epoch == 150: zlr = 2.0
        elif epoch == 225: zlr = 1.0
        train(args, epoch, net, D_net, trainLoader, optimizer, optimizer_D, zlr, device)
        test(args, epoch, net, testLoader, optimizer, device)
        torch.save(net.state_dict(), os.path.join(args.save, 'R.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save, 'D.pt'))
        
    
    net.eval()
    X_test_t, y_test_t = next(iter(testLoader_cor))
    X_test_t, y_test_t = X_test_t.to(device), y_test_t.to(device)