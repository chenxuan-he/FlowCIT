# import pandas as pd

# sim_type=2
# alphas=[-.2, .1, .0, .1, .2]
# n=500

# for alpha in alphas:
#     # Specify the file name
#     file_name = 'results/sim_type' + str(sim_type) + '-alpha-' + str(alpha) + '-n-' + str(n) + '-x-3-y-3-z-3.csv'

#     # Read the CSV file into a DataFrame
#     data = pd.read_csv(file_name, header=None)

#     # Calculate the mean of each column
#     column_means = (data>.05).mean()

#     # Print the column means
#     print(column_means)


import pandas as pd

sim_type = 2
# alphas = [-0.2, 0.1, 0.0, 0.1, 0.2]
# n = 500

alphas = [-0.1, -0.05, 0.0, 0.05, 0.1]
n = 1000

# Create an empty list to store the means
means_list = []

for alpha in alphas:
    # Specify the file name
    file_name = f'results/sim_type{sim_type}-alpha-{alpha}-n-{n}-x-3-y-3-z-3.csv'

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_name, header=None)

    # Calculate the mean of each column
    column_means = (data > 0.05).mean()

    # Append the means to the list along with the alpha value
    means_list.append(column_means)

# Convert the list of means into a DataFrame
means_df = pd.DataFrame(means_list, index=alphas)

# Optionally, you can reset the index to have a column for alpha
means_df.reset_index(inplace=True)
means_df.rename(columns={'index': 'alpha'}, inplace=True)

# Print the resulting DataFrame
print(means_df)