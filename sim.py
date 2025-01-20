from functions import fcit_test, pdc_test, cdc_test
from flow_functions import flow_test
from functions import generate_data
from dgcit_functions import dgcit
import argparse
import os
import torch
import multiprocessing
from torch.multiprocessing import Pool, set_start_method
import concurrent.futures
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--sim_type', type=int, default=0, help='Simulation types, including H_0 and H_1.')
    parser.add_argument('--p', type=int, default=3, help='Dimension of X.')
    parser.add_argument('--q', type=int, default=3, help='Dimension of Y.')
    parser.add_argument('--d', type=int, default=3, help='Dimension of Z.')
    parser.add_argument('--n', type=int, default=100, help='Sample size.')
    parser.add_argument('--alpha', type=float, default=0, help='Deviation under H_1.')
    parser.add_argument('--cores', type=int, default=5, help='Numer of cores for parallel computing.')
    parser.add_argument('--nsim', type=int, default=10, help='Numer of simulations.')
    return parser.parse_args()


def sim(sim_type=0, seed=0, p=3, q=3, d=3, n=100, alpha=.1, batchsize=50, iteration_flow=500, hidden_num=256, lr=5e-3, num_steps=1000, device="cpu"):
    # generate data
    x, y, z = generate_data(sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    # flow test
    dc, p_dc = flow_test(x=x.clone().detach(), y=y.clone().detach(), z=z.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
    _, p_fcit = fcit_test(x, y, z)
    _, p_pdc = pdc_test(x, y, z)
    _, p_cdc = cdc_test(x, y, z)
    p_dgcit = dgcit(x, y, z)
    return dc, p_dc, p_fcit, p_pdc, p_cdc, p_dgcit


def run_simulation(seed, args, device):
    return sim(seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, device=device)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()
    if args.gpu:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nsim = args.nsim

    results = []

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(run_simulation, seed, args, device): seed for seed in range(nsim)}
        
        # As each task completes, print the result
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Seed {seed}: {result}")
            except Exception as exc:
                print(f"Seed {seed} generated an exception: {exc}")

    result_matrix = np.array(results)

    # Write the matrix to a CSV file
    np.savetxt("trial.csv", result_matrix, delimiter=",")


# if __name__=="__main__":
#     args = parse_arguments()
#     if args.gpu:
#         # Set the CUDA_VISIBLE_DEVICES environment variable
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     cores = args.cores
#     nsim = args.nsim

#     for seed in range(nsim):
#         tmp = sim(seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, device=device)
#         print(tmp)
