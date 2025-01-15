# from rpy2.robjects.packages import importr

# # Import ggplot2
# ggplot2 = importr('ggplot2')

# # Create a simple plot using ggplot2 in R
# r_code = """
# library(ggplot2)
# df <- data.frame(x = c(1, 2, 3), y = c(4, 5, 6))
# ggplot(df, aes(x, y)) + geom_point()
# """
# ro.r(r_code)