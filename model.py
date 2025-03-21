from functions_test import fcit_test, pdc_test, cdc_test
from functions_flow import flow_test, flow_test_split
from functions_dgcit import dgcit
from functions_generate_data import read_data

import argparse
import os
import torch
import multiprocessing
from torch.multiprocessing import Pool, set_start_method
import concurrent.futures
import numpy as np
import time
import psutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--cpu', type=str, default="0-10", help='Indices of cpus for parallel computing.')
    parser.add_argument('--model', type=int, default=1, help='Different models in the simulations.')
    parser.add_argument('--sim_type', type=int, default=1, help='Simulation types, like linear or nonlinear.')
    parser.add_argument('--p', type=int, default=3, help='Dimension of X.')
    parser.add_argument('--q', type=int, default=3, help='Dimension of Y.')
    parser.add_argument('--d', type=int, default=3, help='Dimension of Z.')
    parser.add_argument('--n', type=int, default=500, help='Sample size.')
    parser.add_argument('--alpha', type=float, default=.0, help='Deviation under H_1.')
    parser.add_argument('--par_task', type=int, default=5, help='Numer of tasks for parallel computing.')
    parser.add_argument('--nsim', type=int, default=10, help='Numer of simulations.')
    parser.add_argument('--hidden_num', type=int, default=64, help='Hidden dimensions of flow training.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate of flow training.')
    parser.add_argument('--batchsize', type=int, default=50, help='Batchsize of flow training.')
    parser.add_argument('--n_iter', type=int, default=500, help='Iteration of flow training.')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of steps when sampling ODE.')
    parser.add_argument('--dgcit_batchsize', type=int, default=500, help='Batchsize of dgcit.')
    parser.add_argument('--dgcit_n_iter', type=int, default=100, help='Iterations of dgcit.')
    parser.add_argument('--dgcit_k', type=int, default=2, help='K-fold of dgcit.')
    parser.add_argument('--dgcit_j', type=int, default=1000, help='Parameter j of dgcit.')
    parser.add_argument('--dgcit_b', type=int, default=30, help='Parameter b of dgcit.')
    return parser.parse_args()


def sim(model=1, sim_type=1, seed=0, p=3, q=3, d=3, n=500, alpha=.1, batchsize=50, n_iter=500, hidden_num=256, lr=5e-3, num_steps=1000, device="cpu"):
    # generate data
    x, y, z = read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    
    # flow test
    print("\nExecuting flow test.")
    start_time = time.time()
    _, p_dc = flow_test(x=x.clone().detach(), y=y.clone().detach(), z=z.clone().detach(), batchsize=batchsize, n_iter=n_iter, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
    flow_test_time = time.time() - start_time
    print("P-value: "+str(round(p_dc, 2))+". Execution time: "+str(round(flow_test_time, 2)))

    print("\nExecuting fcit test.")
    start_time = time.time()
    _, p_fcit = fcit_test(x, y, z)
    fcit_test_time = time.time() - start_time
    print("P-value: "+str(round(p_fcit, 2))+". Execution time: "+str(round(fcit_test_time, 2)))

    print("\nExecuting cdc test.")
    start_time = time.time()
    _, p_cdc = cdc_test(x, y, z)
    cdc_test_time = time.time() - start_time
    print("P-value: "+str(round(p_cdc, 2))+". Execution time: "+str(round(cdc_test_time, 2)))

    return p_dc, p_fcit, p_cdc, flow_test_time, fcit_test_time, cdc_test_time


def sim_dgcit(model=1, sim_type=1, seed=0, p=3, q=3, d=3, n=500, alpha=.1, batch_size=64, n_iter=1000, k=2, b=30, j=1000):
    x, y, z = read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    print("\nExecuting dgcit test.")
    start_time = time.time()
    print(f"\nTesting with j={j}")
    p_dgcit = dgcit(x, y, z, seed=seed, batch_size=batch_size, n_iter=n_iter, k=k, b=b, j=j)
    dgcit_time = time.time() - start_time
    print("P-value: "+str(round(p_dgcit, 2))+". Execution time: "+str(round(dgcit_time, 2)))
    return p_dgcit, dgcit_time
  

def run_simulation(seed, args, device):
    p_dc, p_fcit, p_cdc, _, _, _ = sim(model=args.model, seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, device=device, hidden_num=args.hidden_num, batchsize=args.batchsize, n_iter=args.n_iter, lr=args.lr, num_steps=args.num_steps)
    p_dgcit, _ = sim_dgcit(model=args.model, seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha,
                           batch_size=args.dgcit_batchsize, n_iter=args.dgcit_n_iter, k=args.dgcit_k, b=args.dgcit_b, j=args.dgcit_j)
    return p_dc, p_fcit, p_cdc, p_dgcit


if __name__ == "__main__":
    args = parse_arguments()
    p_dgcit, dgcit_time = sim_dgcit(model=args.model, seed=0, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha,
                                    batch_size=args.dgcit_batchsize, n_iter=args.dgcit_n_iter, k=args.dgcit_k, b=args.dgcit_b, j=args.dgcit_j)
    print(p_dgcit)


# if __name__ == "__main__":
#     args = parse_arguments()

#     multiprocessing.set_start_method('spawn')
#     p = psutil.Process(os.getpid())
#     start, end = map(int, args.cpu.split('-'))
#     p.cpu_affinity(range(start, end))

#     if args.gpu:
#         # Set the CUDA_VISIBLE_DEVICES environment variable
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     nsim = args.nsim
#     results = []

#     # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
#     with concurrent.futures.ProcessPoolExecutor(max_workers=args.par_task) as executor:
#         # Submit all tasks to the executor
#         futures = {executor.submit(run_simulation, seed, args, device): seed for seed in range(nsim)}
        
#         # As each task completes, print the result
#         for future in concurrent.futures.as_completed(futures):
#             seed = futures[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#                 print(f"Seed {seed}: {result}")
#             except Exception as exc:
#                 print(f"Seed {seed} generated an exception: {exc}")

#     result_matrix = np.array(results)

#     # Write the matrix to a CSV file
#     np.savetxt(f"results/model{args.model}_type{args.sim_type}-alpha-{args.alpha}-n-{args.n}-x-{args.p}-y-{args.q}-z-{args.d}-hidden_num{args.hidden_num}.csv", result_matrix, delimiter=",")

