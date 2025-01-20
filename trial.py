from functions import FCIT, KCI, PDC, CDC
import multiprocessing
from flow_functions import flow_test
from functions import generate_data

def sim(seed=0, p=10, q=10, d=3, n=1000, alpha=.1, batchsize=50, iteration_flow=500, hidden_num=256, lr=5e-3, num_steps=1000):
    # generate data
    (X_H0, Y_H0, Z_H0), (X_H1, Y_H1, Z_H1), (X_H1_2, Y_H1_2, Z_H1_2) = generate_data(alpha=alpha, n=n, p=p, q=q, d=d, seed=seed)
    # flow test
    dc_1, p_1 = flow_test(x=X_H0.clone().detach(), y=Y_H0.clone().detach(), z=Z_H0.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps)
    dc_2, p_2 = flow_test(x=X_H1.clone().detach(), y=Y_H1.clone().detach(), z=Z_H1.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps)
    dc_3, p_3 = flow_test(x=X_H1_2.clone().detach(), y=Y_H1_2.clone().detach(), z=Z_H1_2.clone().detach(), batchsize=batchsize, iteration_flow=iteration_flow, seed=seed, hidden_num=hidden_num, lr=lr, num_steps=num_steps)
    return dc_1, p_1, dc_2, p_2, dc_3, p_3


cores = 5
multiprocessing.freeze_support()
pool = multiprocessing.Pool(cores)
nsim = 5
dc_1, p_1, dc_2, p_2, dc_3, p_3  = zip(*pool.starmap(sim, zip(range(nsim))))
data_list = [dc_1, p_1, dc_2, p_2, dc_3, p_3]
pool.close()
