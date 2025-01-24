import pandas as pd

model=2
sim_type = 1
x = 50
y = 50
z = 100

hidden_num = 32

alphas = [-.8, -.4, 0.0, 0.4, 0.8]
n = 1000

# alphas = [-0.2, -0.1, 0.0, 0.1, 0.2]
# n = 500

# alphas = [-0.1, -0.05, 0.0, 0.05, 0.1]
# n = 1000

# Create an empty list to store the means
means_list = []

for alpha in alphas:
    # Specify the file name
    file_name = f'results/model{model}_type{sim_type}-alpha-{alpha}-n-{n}-x-{x}-y-{y}-z-{z}-hidden_num{hidden_num}.csv'
    # file_name = f'results/sim_type{sim_type}-alpha-{alpha}-n-{n}-x-3-y-3-z-{z}.csv'

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

means_df.columns = ['alpha', 'FlowCIT', 'FCIT', 'PDC', 'CDC', 'DGCIT'] + list(means_df.columns[6:])
means_df_truncated = means_df.iloc[:, :6]

means_df_truncated.to_csv(f'model{model}_type{sim_type}-n-{n}-x-{x}-y-{y}-z-{z}.csv', index=False)
