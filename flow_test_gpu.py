import numpy as np
from dc import distance_correlation, permutation_test
from rect_flow import ConditionalRectifiedFlow, train_conditional_rectified_flow, MLP
import torch
import random
import multiprocessing
import pickle
import os
import psutil
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation script with IV Data Generating Process and SCAD Regression")
    # Add arguments for n, p, q, and first_k_beta with default values
    parser.add_argument('--n', type=int, default=1000, help='Observations.')
    parser.add_argument('--p', type=int, default=10, help='Dimension of X')
    parser.add_argument('--q', type=int, default=10, help='Dimension of Y')
    parser.add_argument('--d', type=int, default=10, help='Dimension of Z.')
    parser.add_argument('--nsim', type=int, default=10,
                        help='Number of simulations.')
    parser.add_argument('--cores', type=int, default=5,
                        help="Number of parallel cores.")
    # Parse arguments
    return parser.parse_args()

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


def flow_test(x, y, z, batchsize=50, iteration_flow=500, hidden_num=256, lr=5e-3, num_steps=1000, seed=0, device="cpu"):
    '''
    num_steps: sampling ode steps (Euler's method).
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    n, p = x.shape
    _, q = y.shape
    _, d = z.shape
    eps1 = torch.randn((n,p))
    x1_pairs = [x, eps1, z.detach().clone()]
    rectified_flow_1 = ConditionalRectifiedFlow(model=MLP(input_dim=p+d+1, output_dim=p, hidden_num=hidden_num), num_steps=num_steps, device=device)
    optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=lr)
    rectified_flow_1, loss_curve1 = train_conditional_rectified_flow(rectified_flow_1, optimizer, x1_pairs, batchsize, iteration_flow, device=device)
    eps1_pred = rectified_flow_1.sample_conditional_ode(x, z, device=device)[-1]
    eps2 = torch.randn(size=(n,q))
    y1_pairs = [y, eps2, z.detach().clone()]
    rectified_flow_2 = ConditionalRectifiedFlow(model=MLP(input_dim=q+d+1, output_dim=q, hidden_num=hidden_num), num_steps=num_steps, device=device)
    optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=lr)
    rectified_flow_2, loss_curve2 = train_conditional_rectified_flow(rectified_flow_2, optimizer, y1_pairs, batchsize, iteration_flow, device=device)
    eps2_pred = rectified_flow_2.sample_conditional_ode(y, z, device=device)[-1]
    # perform test
    dc, dc_p = permutation_test(eps1_pred, eps2_pred)
    return dc, dc_p


def sim(seed=0, p=10, q=10, d=3, n=1000, alpha=.1, batchsize=50, iteration_flow=500, hidden_num=256, lr=5e-3, num_steps=100, device="cpu"):
    # generate data
    (X_H0, Y_H0, Z_H0), (X_H1, Y_H1, Z_H1), (X_H1_2, Y_H1_2, Z_H1_2) = generate_data(alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    # flow test
    dc_1, p_1 = flow_test(x=X_H0.clone().detach(), y=Y_H0.clone().detach(), z=Z_H0.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
    dc_2, p_2 = flow_test(x=X_H1.clone().detach(), y=Y_H1.clone().detach(), z=Z_H1.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
    dc_3, p_3 = flow_test(x=X_H1_2.clone().detach(), y=Y_H1_2.clone().detach(), z=Z_H1_2.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
    return dc_1, p_1, dc_2, p_2, dc_3, p_3


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    cpu_cores_to_use = os.cpu_count()*9 // 10
    p = psutil.Process()
    p.cpu_affinity(list(range(cpu_cores_to_use))) 

    cores = 5
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(cores)
        
    args = parse_args()
    nsim = args.nsim
    device_ids = [f'cuda:{i % min(torch.cuda.device_count(), cores)}' if torch.cuda.is_available() else 'cpu' for i in range(nsim)]
    # device_ids = [f'cuda:1' if torch.cuda.is_available() else 'cpu']
    results = pool.starmap(sim, [(i, args.p, args.q, args.d, args.n, 0.1, 50, 1000, 128, 5e-3, 1000, device_ids[i]) for i in range(nsim)])
    pool.close()

    dc_1, p_1, dc_2, p_2, dc_3, p_3 = zip(*results)
    data_list = [dc_1, p_1, dc_2, p_2, dc_3, p_3]
    
    # Save data_list to a file
    with open('flow_test_res/tmp_'+str(args.p) + '_'+ str(args.q) + '_' + str(args.d) + 'simulation_results.pkl', 'wb') as f:
        pickle.dump(data_list, f)