# ---- Load the .rda file and print the results ----
rm(list = ls())
library(CondIndTests)
library(doParallel)
library(foreach)

source("CLZ/MI_test_functions.R")
source("functions_generate_data.R")

library(optparse)

## ---- Accept parameters from terminal command ----
option_list <- list(
  make_option("--model", type = "integer", default = 1, help = "Models."),
  make_option("--sim_type", type = "integer", default = 1, help = "Scenarios."),
  make_option("--alpha", type = "double", default = .0, help = "Deviation of H0 or H1."),
  make_option("--n", type = "integer", default = 500, help = "Sample size."),
  make_option("--p", type = "integer", default = 1, help = "Dimension of X."),
  make_option("--q", type = "integer", default = 1, help = "Dimension of Y."),
  make_option("--d", type = "integer", default = 3, help = "Dimension of Z."),
  make_option("--n_sim", type = "integer", default = 200, help = "Number of simulations."),
  make_option("--n_cpu", type = "integer", default = 50, help = "Max number of cpu to be used in parallel computing."),
  make_option("--bandwidth", type = "double", default = 1.2, help = "A constant before the bandwidth to control $H_0$.")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

model <- opt$model
sim_type <- opt$sim_type
alpha <- opt$alpha
n <- opt$n
p <- opt$p
q <- opt$q
d <- opt$d
n_sim <- opt$n_sim
n_cpu <- opt$n_cpu
bandwidth <- opt$bandwidth

# ---- Start testing: CLZ, KCI ----
old <- Sys.time() # get start time

rdaname = paste0("CLZ/test_results_model", model, "_simtype", sim_type, "_alpha", sprintf("%.1f", alpha), "_n", n, "_p", p, "_q", q, "_d", d, "_bandwidth", bandwidth, ".rda")

load(rdaname)
print(colMeans(pvalues<.05))
