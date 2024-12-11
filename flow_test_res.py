import pandas as pd

# Load a DataFrame from a .pkl file
df = pd.read_pickle('flow_test_res/sims_3_3_2simulation_results.pkl')

sum(1 for value in df[1] if value > 0.05)

sum(1 for value in df[3] if value > 0.05)

sum(1 for value in df[5] if value > 0.05)