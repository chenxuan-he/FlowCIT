import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import argparse

def parse_alpha(alpha_str):
    return [float(a) for a in alpha_str.split(',')]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--n', type=int, default=500, help='Sample size.')
    parser.add_argument('--alphas', type=parse_alpha, default="0,0.05,0.1,0.15,0.2", help='List of alphas.')
    parser.add_argument('--model', type=int, default=1, help='Different models in the simulations.')
    parser.add_argument('--sim_type', type=int, default=1, help='Different simulation types.')
    parser.add_argument('--legend', type=int, default=1, help='Show legend or not.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sim_type = args.sim_type
    model = args.model
    alphas = args.alphas
    n = args.n
    legend = args.legend
    data = pd.read_csv(f'model{model}_simtype{sim_type}-n-{n}.csv')

    # Create the plot
    plt.figure(figsize=(4, 3))
    plt.plot(data['alpha'], data['FlowCIT'], label='FlowCIT', marker='o', linestyle='-', color='b')
    plt.plot(data['alpha'], data['KCI'], label='KCI', marker='D', linestyle='-.', color='c')
    plt.plot(data['alpha'], data['CDC'], label='CDC', marker='*', linestyle='-', color='m')
    plt.plot(data['alpha'], data['CCIT'], label='CCIT', marker='x', linestyle='--', color='y')
    plt.plot(data['alpha'], data['FCIT'], label='FCIT', marker='s', linestyle='--', color='g')
    plt.plot(data['alpha'], data['CLZ'], label='CLZ', marker='^', linestyle=':', color='r')

    # Add a horizontal line at 0.05
    plt.axhline(y=0.05, color='black', linestyle='-')

    # Customize the plot
    # plt.xlabel('Alpha')
    # plt.ylabel('Power/Size')
    # plt.title('Power/Size Comparison Plot')
    if legend:
        plt.legend()
        plt.legend(loc='upper right', bbox_to_anchor=(0.4, 1))
    plt.grid(True)

    # Adjust the x-axis labels
    plt.xticks(alphas)

    # Show the plot
    plt.savefig(f"plots/model{model}_type{sim_type}-n-{n}.pdf", bbox_inches='tight')
    plt.close()
