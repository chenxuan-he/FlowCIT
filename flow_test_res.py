import pandas as pd

# Specify the file name
file_name = 'results/sim_type0-alpha-0.5-n-500-x-3-y-3-z-3.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_name, header=None)

# Calculate the mean of each column
column_means = (data>.05).mean()

# Print the column means
print("Column Means:")
print(column_means)