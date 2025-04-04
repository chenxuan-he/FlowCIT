from functions_test import fcit_test, cdc_test
from functions_flow import flow_test
from functions_generate_data import read_data

import random
import argparse
import os
import torch
import multiprocessing
from torch.multiprocessing import Pool, set_start_method
import concurrent.futures
import numpy as np
import time
import psutil
from CCIT import CCIT

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
    parser.add_argument('--FlowCIT', type=int, default=1, help='Implement FlowCIT or not.')
    parser.add_argument('--FCIT', type=int, default=1, help='Implement FCIT or not.')
    parser.add_argument('--CDC', type=int, default=1, help='Implement CDC or not.')
    parser.add_argument('--CCIT', type=int, default=1, help='Implement CCIT or not.')
    parser.add_argument('--demo', type=int, default=0, help='Run once for demonstration.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for implementation.')
    parser.add_argument('--FlowCIT_method', type=str, default="DC", help="Method to compute transformed pair's correlation. Can be DC (distance correlation) or IPC (Improved projection correlation).")
    parser.add_argument('--FlowCIT_permutation', type=int, default=1, help="For FlowCIT: permutation or not.")
    return parser.parse_args()


def sim(permutation=1, method="DC", model=1, sim_type=1, seed=0, p=3, q=3, d=3, n=500, alpha=.1, batchsize=50, n_iter=500, hidden_num=256, lr=5e-3, num_steps=1000, device="cpu", FlowCIT=1, FCIT=1, CDC=1, CCIT_exe=1):
    # generate data
    x, y, z = read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    
    if FlowCIT:
        # flow test
        print("\nExecuting flow test.")
        start_time = time.time()
        _, p_dc = flow_test(permutation=permutation, method=method, x=x.clone().detach(), y=y.clone().detach(), z=z.clone().detach(), batchsize=batchsize, n_iter=n_iter, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps, device=device)
        flow_test_time = time.time() - start_time
        print("P-value: "+str(round(p_dc, 2))+". Execution time: "+str(round(flow_test_time, 2)))
    else:
        flow_test_time = 0
        p_dc = 0
    
    if FCIT:
        print("\nExecuting fcit test.")
        start_time = time.time()
        _, p_fcit = fcit_test(x, y, z)
        fcit_test_time = time.time() - start_time
        print("P-value: "+str(round(p_fcit, 2))+". Execution time: "+str(round(fcit_test_time, 2)))
    else:
        fcit_test_time = 0
        p_fcit = 0

    if CDC:
        print("\nExecuting cdc test.")
        start_time = time.time()
        _, p_cdc = cdc_test(x, y, z)
        cdc_test_time = time.time() - start_time
        print("P-value: "+str(round(p_cdc, 2))+". Execution time: "+str(round(cdc_test_time, 2)))
    else:
        cdc_test_time = 0
        p_cdc = 0

    if CCIT_exe:
        print("\nExecuting ccit test.")
        start_time = time.time()
        p_ccit = CCIT.CCIT(x.numpy(), y.numpy(), z.numpy())
        ccit_test_time = time.time() - start_time
        print("P-value: "+str(round(p_ccit, 2))+". Execution time: "+str(round(ccit_test_time, 2)))
    else:
        ccit_test_time = 0
        p_ccit = 0

    return p_dc, p_fcit, p_cdc, p_ccit, flow_test_time, fcit_test_time, cdc_test_time, ccit_test_time

def run_simulation(seed, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    p_dc, p_fcit, p_cdc, p_ccit, _, _, _, _ = sim(permutation=args.FlowCIT_permutation, method=args.FlowCIT_method, model=args.model, seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, device=device, hidden_num=args.hidden_num, batchsize=args.batchsize, n_iter=args.n_iter, lr=args.lr, num_steps=args.num_steps, FlowCIT=args.FlowCIT, FCIT=args.FCIT, CDC=args.CDC, CCIT_exe=args.CCIT)
    return p_dc, p_fcit, p_cdc, p_ccit


# A parallel version
if __name__ == "__main__":
    args = parse_arguments()
    if args.demo:
        if args.gpu:
            # Set the CUDA_VISIBLE_DEVICES environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        run_simulation(seed=args.seed, args=args, device=device)
    elif not args.demo:
        multiprocessing.set_start_method('spawn')
        p = psutil.Process(os.getpid())
        start, end = map(int, args.cpu.split('-'))
        p.cpu_affinity(range(start, end))

        if args.gpu:
            # Set the CUDA_VISIBLE_DEVICES environment variable
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nsim = args.nsim
        results = []

        # Use ThreadPoolExecutor for I/O-bound tasks or ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.par_task) as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(run_simulation, seed+args.seed, args, device): seed for seed in range(nsim)}
            
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
        np.savetxt(f"results/model{args.model}_type{args.sim_type}-alpha-{args.alpha}-n-{args.n}-x-{args.p}-y-{args.q}-z-{args.d}-hidden_num{args.hidden_num}.csv", result_matrix, delimiter=",")

