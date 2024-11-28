import pandas as pd

# Load a DataFrame from a .pkl file
df = pd.read_pickle('flow_test_res/10_10_10simulation_results.pkl')

print(df)