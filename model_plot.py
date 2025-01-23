import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

sim_type = 1
model = 1
z = 3
x = 3
y = 3 

# alphas = [-0.2, -0.1, 0.0, 0.1, 0.2]
# n = 500

alphas = [-0.1, -0.05, 0.0, 0.05, 0.1]
n = 1000

data = pd.read_csv(f'model{model}_type{sim_type}-n-{n}-x-3-y-3-z-{z}.csv')

# Create the plot
plt.figure(figsize=(4, 3))
plt.plot(data['alpha'], data['FlowCIT'], label='FlowCIT', marker='o', linestyle='-', color='blue')
plt.plot(data['alpha'], data['FCIT'], label='FCIT', marker='s', linestyle='--', color='red')
plt.plot(data['alpha'], data['PDC'], label='PDC', marker='^', linestyle='-.', color='green')
plt.plot(data['alpha'], data['CDC'], label='CDC', marker='*', linestyle=':', color='purple')
plt.plot(data['alpha'], data['DGCIT'], label='DGCIT', marker='D', linestyle=':', color='orange')

# Add a horizontal line at 0.05
plt.axhline(y=0.05, color='black', linestyle='--')

# Customize the plot
# plt.xlabel('Alpha')
# plt.ylabel('Power/Size')
# plt.title('Power/Size Comparison Plot')
# plt.legend()
# plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
plt.grid(True)

# Adjust the x-axis labels
plt.xticks(alphas)

# Show the plot
plt.savefig(f"plots/model{model}_type{sim_type}-n-{n}-x-{x}-y-{y}-z-{z}.pdf", bbox_inches='tight')
plt.close()
