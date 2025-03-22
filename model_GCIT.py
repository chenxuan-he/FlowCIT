# from functions_dgcit import dgcit
from functions_generate_data import read_data
from functions_gcit import GCIT

import random
import argparse
import os
import tensorflow as tf
import multiprocessing
from torch.multiprocessing import Pool, set_start_method
import concurrent.futures
import numpy as np
import time
import psutil


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--cpu', type=str, default="0-50", help='Indices of cpus for parallel computing.')
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
    parser.add_argument('--test_prop', type=float, default=.1, help='Porportion for test to control the size.')
    return parser.parse_args()


# def sim_dgcit(model=1, sim_type=1, seed=0, p=3, q=3, d=3, n=500, alpha=.1, batch_size=64, n_iter=1000, k=2, b=30, M=500, j=1000):
#     x, y, z = read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
#     print("\nExecuting dgcit test.")
#     start_time = time.time()
#     p_dgcit = dgcit(x=x, y=y, z=z, seed=seed, batch_size=batch_size, n_iter=n_iter, k=k, b=b, M=M, j=j)
#     dgcit_time = time.time() - start_time
#     print("P-value: "+str(round(p_dgcit, 2))+". Execution time: "+str(round(dgcit_time, 2)))
#     return p_dgcit, dgcit_time


def sim_gcit(model=1, sim_type=1, seed=0, p=3, q=3, d=3, n=500, alpha=.1, test_prop=.3):
    x, y, z = read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    print("\nExecuting gcit test.")
    start_time = time.time()
    p_gcit = GCIT(x=x, y=y, z=z, statistic = "rdc", lamda = 50, normalize=True, verbose=False, n_iter=1000, debug=False, test_prop=test_prop)
    gcit_time = time.time() - start_time
    print("P-value: "+str(round(p_gcit, 2))+". Execution time: "+str(round(gcit_time, 2)))
    return p_gcit, gcit_time


def run_simulation(seed, args):
    random.seed(seed)
    np.random.seed(seed)
    # p_dgcit, _ = sim_dgcit(model=args.model, seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha,
    #                        batch_size=args.dgcit_batchsize, n_iter=args.dgcit_n_iter, k=args.dgcit_k, b=args.dgcit_b, M=args.dgcit_M, j=args.dgcit_j)

    tf.random.set_random_seed(seed)
    p_gcit, _ = sim_gcit(model=args.model, seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, test_prop=args.test_prop)
    return p_gcit


# # A demo
# if __name__ == "__main__":
#     args = parse_arguments()
#     run_simulation(seed=0, args=args, device="cpu")


# A parallel version
if __name__ == "__main__":
    args = parse_arguments()

    multiprocessing.set_start_method('spawn')
    p = psutil.Process(os.getpid())
    start, end = map(int, args.cpu.split('-'))
    p.cpu_affinity(range(start, end))

    nsim = args.nsim
    results = []

    # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.par_task) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(run_simulation, seed, args): seed for seed in range(nsim)}
        
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
    np.savetxt(f"results/model{args.model}_type{args.sim_type}-alpha-{args.alpha}-n-{args.n}-x-{args.p}-y-{args.q}-z-{args.d}-GCIT-test_prop{args.test_prop}.csv", result_matrix, delimiter=",")
