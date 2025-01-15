import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
# Import ggplot2
ggplot2 = importr('ggplot2')

# Create a simple plot using ggplot2 in R
r_code = """
df <- rnorm(100)
"""
ro.r(r_code)
r_data = ro.r['df']

np.array(r_data)
