# ---- Preparation ----
## ---- Load packages and functions ----
rm(list = ls())
library(energy)
library(bnlearn)
library(Rcpp)
library(cdcsis)
library(CondIndTests)
library(praznik)
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
## ---- Parse the command line arguments ----
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

# ---- Start testing: CLZ, CDC, KCI, SW ----
old <- Sys.time() # get start time

B = 100
rdaname = paste0("CLZ/test_results_model", model, "_simtype", sim_type, "_alpha", sprintf("%.1f", alpha), "_n", n, "_p", p, "_q", q, "_d", d, "_bandwidth", bandwidth, ".rda")

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


## ---- Independence Test ----
num_cores <- min(detectCores() - 1, n_cpu)
cl <- makeCluster(num_cores)
registerDoParallel(cl)

pvalues <- foreach(i = 0:(n_sim-1), .combine = rbind, .packages = c("CondIndTests")) %dopar%{
  data_list <- read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=i)
  X <- data_list$X
  Y <- data_list$Y
  Z <- data_list$Z
  
  h = bandwidth*1.06*(4/(3*n))^{1/(2*(d+min(p,q)-1)-1)}
  
  # pval1 corresponds to the proposed CIT of our paper
  test.stat =  CI.multiXYZ.test(X, Y, Z, h)
  pval1 = mean(boots.stat>test.stat)
  
  # # pval3 corresponds to CDC 
  # hdc = 1*1.06*sd(Z[,1])*(4/(3*n))^{1/(1+4)}
  # pval2 = pdcor.test(X, Y, Z, R = 100)$p.value
  # pval3 = cdcov.test(X, Y, Z,width = hdc)$p.value
  
  # # pval4 corresponds to CMI
  # cmi = cmiScores(X, Y, Z)
  # boot.vec = rep(0, B)
  # for (jj in 1:B) {
  #   #cat(jj, "\r")
  #   Xnew = local.boots.index(Z, X)
  #   boot.vec[jj] = cmiScores(Xnew, Y, Z)
  # }
  # pval4 = mean(boot.vec>cmi)
  
  # pval5 corresponds to KCI
  pval5 = KCI(X, Y, Z)$pvalue
  return(c(pval1, pval5))
}
stopCluster(cl)

# ### Report size and power
# print("alpha is 0.05")
# mean(pval1<0.05)
# mean(pval2<0.05)
# mean(pval3<0.05)
# mean(pval4<0.05)
# mean(pval5<0.05)

# print("alpha is 0.1")
# mean(pval1<0.1)
# mean(pval2<0.1)
# mean(pval3<0.1)
# mean(pval4<0.1)
# mean(pval5<0.1)

# a1 = c(mean(pval1<0.05), mean(pval2<0.05), mean(pval3<0.05), mean(pval4<0.05), mean(pval5<0.05))
# a2 = c(mean(pval1<0.1), mean(pval2<0.1), mean(pval3<0.1), mean(pval4<0.1), mean(pval5<0.1))

# aa = rbind(a1, a2)
save(pvalues, file = rdaname)

# # print elapsed time
# new <- Sys.time() - old # calculate difference
# print(new) # print in nice format
