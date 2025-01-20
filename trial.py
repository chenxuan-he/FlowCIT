from functions import fcit_test, pdc_test, cdc_test
import multiprocessing
from flow_functions import flow_test
from functions import generate_data

def sim(sim_type=0, seed=0, p=3, q=3, d=3, n=100, alpha=.1, batchsize=50, iteration_flow=500, hidden_num=256, lr=5e-3, num_steps=1000):
    # generate data
    x, y, z = generate_data(sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    # flow test
    dc, p_dc = flow_test(x=x.clone().detach(), y=y.clone().detach(), z=z.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps)
    fcit, p_fcit = fcit_test(x, y, z)
    pdc, p_pdc = pdc_test(x, y, z)
    cdc, p_cdc = cdc_test(x, y, z)
    return dc, p_dc, fcit, p_fcit, pdc, p_pdc, cdc, p_cdc


cores = 5
multiprocessing.freeze_support()
pool = multiprocessing.Pool(cores)
nsim = 5
dc_1, p_1, dc_2, p_2, dc_3, p_3  = zip(*pool.starmap(sim, zip(range(nsim))))
data_list = [dc_1, p_1, dc_2, p_2, dc_3, p_3]
pool.close()
