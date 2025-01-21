from sim import sim
from functions import fcit_test, pdc_test, cdc_test
from flow_functions import flow_test, flow_test_split
from functions import generate_data
from dgcit_functions import dgcit
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
    parser.add_argument('--gpu', type=str, default="0,1", help='Comma-separated list of GPU indices to use.')
    parser.add_argument('--cpu', type=str, default="0-10", help='Numer of maximum numer of cpus for parallel computing.')
    parser.add_argument('--sim_type', type=int, default=0, help='Simulation types, including H_0 and H_1.')
    parser.add_argument('--p', type=int, default=3, help='Dimension of X.')
    parser.add_argument('--q', type=int, default=3, help='Dimension of Y.')
    parser.add_argument('--d', type=int, default=3, help='Dimension of Z.')
    parser.add_argument('--n', type=int, default=100, help='Sample size.')
    parser.add_argument('--alpha', type=float, default=0, help='Deviation under H_1.')
    parser.add_argument('--par_task', type=int, default=5, help='Numer of tasks for parallel computing.')
    parser.add_argument('--nsim', type=int, default=10, help='Numer of simulations.')
    return parser.parse_args()


def run_simulation(seed, args, device):
    print("\nStarting running simulations.")
    return sim(seed=seed, sim_type=args.sim_type, p=args.p, q=args.q, d=args.d, n=args.n, alpha=args.alpha, device=device)


if __name__ == "__main__":
    args = parse_arguments()
    p = psutil.Process(os.getpid())
    start, end = map(int, args.cpu.split('-'))
    p.cpu_affinity(range(start, end))

    if args.gpu:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nsim = args.nsim

    results = []

    tmp = run_simulation(seed=0, args=args, device=device)

    print(tmp)