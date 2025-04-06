# ---- Preparation ----
## ---- Load packages and functions ----
rm(list = ls())
library(CondIndTests)
library(doParallel)
library(foreach)

source("CLZ/MI_test_functions.R")
source("functions_generate_data.R")

library(optparse)

## ---- Accept parameters from terminal command ----
option_list <- list(
  make_option("--bandwidth", type = "double", default = 1.2, help = "A constant before the bandwidth to control $H_0$."),
  make_option("--KCI", type = "integer", default = 1, help = "Perform KCI test or not."),
  make_option("--CLZ", type = "integer", default = 1, help = "Perform CLZ test or not."),
  make_option("--postfix", type = "character", default = "pca", help = "Method of dimension reduction."),
  make_option("--n_cpu", type = "integer", default = 50, help = "Number of cores for parallel computing.")
)
## ---- Parse the command line arguments ----
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

postfix = opt$postfix
bandwidth <- opt$bandwidth
KCI <- opt$KCI
CLZ <- opt$CLZ
n_cpu <- opt$n_cpu

# ---- Start testing: CLZ, KCI ----
old <- Sys.time() # get start time

# Get data
X <- as.matrix(read.csv(paste0("data/x_", postfix, ".csv")))
Y <- as.matrix(read.csv(paste0("data/y_", postfix, ".csv")))
Z <- as.matrix(read.csv(paste0("data/z_", postfix, ".csv")))

n <- dim(X)[1]
p <- dim(X)[2]
q <- dim(Y)[2]
d <- dim(Z)[2]

if(CLZ){
  B = 100
  rdaname = paste0("CLZ/test_results_model_", postfix, "_n", n, "_p", p, "_q", q, "_d", d, "_bandwidth", bandwidth, ".rda")
  
  ## ---- For CLZ: bootstrap null data ----
  # get the null distribution (generate x y z with the same sample size and dimensions as the original data)
  filename = paste0("CLZ/test_bootstrap_data_n", n, "_p", p, "_q", q, "_d", d, "_bandwidth", bandwidth, ".rda")
  if(file.exists(filename)) {load(filename);print("Already have bootstrap file!")} else {
    #### Bootstrap the null distribution
    print("Now start bootstrap!")
    
    # CLZ theorem require that: $nh^{2(d+p+1)} \to \infty$; $nh^{2(d+q+1)} \to \infty$; we select $h \asymp \max(1/(2(d+p-1)-1), 1/(2(d+q-1)-1))$
    h = bandwidth*1.06*(4/(3*n))^{1/(2*(d+min(p,q)-1)-1)}
    boots.time = 1000
    
    # Set up parallel backend
    num_cores <- min(detectCores() - 1, n_cpu)
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    
    # Run bootstrap in parallel
    boots.stat <- foreach(i = 1:boots.time, .combine = c) %dopar% {
      if (i %% 100 == 0) print(i)
      X <- matrix(rnorm(n * p), n, p)
      Y <- matrix(rnorm(n * q), n, q)
      Z <- matrix(rnorm(n * d), n, d)
      CI.multiXYZ.test(X, Y, Z, h)
    }
    
    # Stop parallel cluster
    stopCluster(cl)
    save(boots.stat, file = filename)
    print("Finished bootstrap!")
  }
}


## ---- Independence Test ----
if(CLZ){
  h = bandwidth*1.06*(4/(3*n))^{1/(2*(d+min(p,q)-1)-1)}
  
  # pval1 corresponds to the proposed CIT of our paper
  test.stat =  CI.multiXYZ.test(X, Y, Z, h)
  pval1 = mean(boots.stat>test.stat)
}else{
  pval1 = -1
}
if(KCI){
  # pval5 corresponds to KCI
  pval5 = KCI(X, Y, Z)$pvalue
}else{
  pval5 = -1
}
print(c(pval1, pval5))
new <- Sys.time() - old # calculate difference
print(new) # print in nice format