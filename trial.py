from functions import fcit_test, pdc_test, cdc_test
import multiprocessing
from flow_functions import flow_test
from functions import generate_data
from dgcit_functions import dgcit
import argparse
import os
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--gpu', type=str, default="2", help='Comma-separated list of GPU indices to use.')
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


if __name__=="__main__":
    args = parse_arguments()

    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    if args.gpu:
        # Set the CUDA_VISIBLE_DEVICES environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tmp = sim(device=device)
    print(tmp)

# cores = 5
# multiprocessing.freeze_support()
# pool = multiprocessing.Pool(cores)
# nsim = 5
# dc_1, p_1, dc_2, p_2, dc_3, p_3  = zip(*pool.starmap(sim, zip(range(nsim))))
# data_list = [dc_1, p_1, dc_2, p_2, dc_3, p_3]
# pool.close()
