import pandas as pd

sim_type=2
alpha=.0
n=1000
# Specify the file name
file_name = 'results/sim_type' + str(sim_type) + '-alpha-' + str(alpha) + '-n-' + str(n) + '-x-3-y-3-z-3.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_name, header=None)

# Calculate the mean of each column
column_means = (data>.05).mean()

# Print the column means
print(column_means)