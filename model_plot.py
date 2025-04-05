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
    parser.add_argument('--FlowCIT', type=int, default=1)
    parser.add_argument('--KCI', type=int, default=1)
    parser.add_argument('--CDC', type=int, default=1)
    parser.add_argument('--CCIT', type=int, default=1)
    parser.add_argument('--FCIT', type=int, default=1)
    parser.add_argument('--CLZ', type=int, default=1)

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
    if args.FlowCIT:
        plt.plot(data['alpha'], data['FlowCIT'], label='FlowCIT', marker='o', linestyle='-', color='#C44E52', markersize=4, linewidth=1.2, alpha=1)
    if args.KCI:
        plt.plot(data['alpha'], data['KCI'], label='KCI', marker='D', linestyle='-', color='#4C72B0', markersize=4, linewidth=1.2, alpha=1)
    if args.CDC:
        plt.plot(data['alpha'], data['CDC'], label='CDC', marker='^', linestyle='-', color='#EEB021', markersize=4, linewidth=1.2, alpha=1)
    if args.CCIT:
        plt.plot(data['alpha'], data['CCIT'], label='CCIT', marker='x', linestyle='-', color='#55A868', markersize=4, linewidth=1.2, alpha=1)
    if args.FCIT:
        plt.plot(data['alpha'], data['FCIT'], label='FCIT', marker='1', linestyle='-', color='#7A7C7E', markersize=6, linewidth=1.2, alpha=0.6)
    if args.CLZ:
        plt.plot(data['alpha'], data['CLZ'], label='CLZ', marker='|', linestyle='-', color='#8172B3', markersize=6, linewidth=1.2, alpha=1)

    # Add a horizontal line at 0.05
    plt.axhline(y=0.05, color='grey', linestyle='--', alpha=.5, linewidth=.6)

    # Customize the plot
    # plt.xlabel('Alpha')
    # plt.ylabel('Power/Size')
    # plt.title('Power/Size Comparison Plot')
    if legend:
        plt.legend(loc='upper right', 
                   bbox_to_anchor=(0.3, 1),
                   fontsize=7.5,        # Legend text size
                   markerscale=1,     # Scale legend markers
        )

    plt.grid(False)

    # Adjust the x-axis labels
    plt.xticks(alphas)

    plt.ylim(-0.02, 1.02)                    # Set y-axis limits
    plt.yticks([0, 0.05, 0.25, 0.50, 0.75, 1.00])  # Set y-axis ticks

    # Show the plot
    plt.savefig(f"plots/model{model}_type{sim_type}-n-{n}.pdf", bbox_inches='tight')
    plt.close()
