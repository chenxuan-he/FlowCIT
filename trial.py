import numpy as np
from functions_IPC import IPC

# Define the shape of the vectors
shape = (200, 200)

# Generate the first 200x200 normal vector
vector1 = np.random.normal(loc=0.0, scale=1.0, size=shape)

# Generate the second 200x200 normal vector
vector2 = np.random.normal(loc=0.0, scale=1.0, size=shape)

IPC(vector1, vector2)