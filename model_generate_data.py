# since tests are performed separately on R and python, here we generate data and write to local .csv file
import argparse
from functions_generate_data import generate_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--model', type=int, default=1, help='Different models in the simulations.')
    parser.add_argument('--sim_type', type=int, default=0, help='Simulation types, like linear or nonlinear.')
    parser.add_argument('--p', type=int, default=3, help='Dimension of X.')
    parser.add_argument('--q', type=int, default=3, help='Dimension of Y.')
    parser.add_argument('--d', type=int, default=3, help='Dimension of Z.')
    parser.add_argument('--n', type=int, default=100, help='Sample size.')
    parser.add_argument('--alpha', type=float, default=0, help='Deviation under H_1.')
    parser.add_argument('--nsim', type=int, default=10, help='Numer of simulations.')
    parser.add_argument('--seed', type=int, default=0, help='Numer of seed.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    for i in range(args.nsim):
        generate_data(model=args.model, sim_type=args.sim_type, 
                      n=args.n, p=args.p, q=args.q, d=args.d, 
                      alpha=args.alpha, seed=i+args.seed)