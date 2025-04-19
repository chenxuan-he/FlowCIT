import pandas as pd
import argparse

def parse_alpha(alpha_str):
    return [float(a) for a in alpha_str.split(',')]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process GPU indices.')
    parser.add_argument('--n', type=int, default=500, help='Sample size.')
    parser.add_argument('--p', type=int, default=3, help='Dimension of X.')
    parser.add_argument('--q', type=int, default=3, help='Dimension of Y.')
    parser.add_argument('--d', type=int, default=3, help='Dimension of Z.')
    parser.add_argument('--alphas', type=parse_alpha, default="0,0.05,0.1,0.15,0.2", help='List of alphas.')
    parser.add_argument('--model', type=int, default=1, help='Different models in the simulations.')
    parser.add_argument('--sim_type', type=int, default=1, help='Different simulation types.')
    parser.add_argument('--hidden_num', type=int, default=64, help='Hidden dimension of FlowCIT.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    n = args.n
    p = args.p
    q = args.q
    d = args.d
    alphas = args.alphas
    model = args.model
    sim_type = args.sim_type
    hidden_num = args.hidden_num

    # Create an empty list to store the means
    means_list = []

    for alpha in alphas:
        # Specify the file name
        file_name = f'results/model{model}_type{sim_type}-alpha-{alpha}-n-{n}-x-{p}-y-{q}-z-{d}-hidden_num{hidden_num}.csv'

        # Read the CSV file into a DataFrame
        data = pd.read_csv(file_name, header=None)

        # Calculate the mean of each column
        column_means = (data < 0.05).mean()

        # Append the means to the list along with the alpha value
        means_list.append(column_means)

    # Convert the list of means into a DataFrame
    means_df = pd.DataFrame(means_list, index=alphas)

    # Optionally, you can reset the index to have a column for alpha
    means_df.reset_index(inplace=True)
    means_df.rename(columns={'index': 'alpha'}, inplace=True)

    means_df.columns = ['alpha', 'FlowCIT', 'FCIT', 'CDC', 'CCIT'] + list(means_df.columns[5:])
    means_df_truncated = means_df.iloc[:, :5]

    means_df_truncated.to_csv(f'model{model}_simtype{sim_type}-n-{n}-x-{p}-y-{q}-z-{d}.csv', index=False)
