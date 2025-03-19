# ---- Preparation ----
## ---- Load packages and functions ----
rm(list = ls())
set.seed(1234)
library(energy)
library(bnlearn)
library(Rcpp)
library(cdcsis)
library(CondIndTests)
library(praznik)

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
  make_option("--n_sim", type = "integer", default = 200, help = "Number of simulations.")
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

# ---- Start testing: CLZ, CDC, KCI, SW ----
old <- Sys.time() # get start time

B = 100
rdaname = paste0("CLZ/test_results_model", model, "_simtype", sim_type, "_alpha", sprintf("%.1f", alpha), "_n", n, "_p", p, "_q", q, "_d", d, ".rda")

## ---- For CLZ: bootstrap null data ----
# get the null distribution (generate x y z with the same sample size and dimensions as the original data)
filename = paste0("CLZ/test_bootstrap_data_n", n, "_p", p, "_q", q, "_d", d, ".rda")
if(file.exists(filename)) {load(filename);print("Already have bootstrap file!")} else {
  #### Bootstrap the null distribution
  print("Now start bootstrap!")
  h = 1.2*1.06*1*(4/(3*n))^{1/(1+4)}
  boots.time = 1000
  boots.stat = rep(0, boots.time)
  for (i in 1:boots.time) {
    if(i%%100==0) {print(i)}
    X = matrix(rnorm(n*p), n, p)
    Y = matrix(rnorm(n*q), n, q)
    Z = matrix(rnorm(n*d), n, d)
    boots.stat[i] = CI.multiXYZ.test(X, Y, Z, h)
  }
  save(boots.stat, file = filename)
  print("Finished bootstrap!")
}

## ---- Independence Test ----
pval1 <- pval2 <- pval3 <- pval4 <- pval5 <- rep(0, n_sim)

for (i in 0:(n_sim-1)) {
  cat(i, "\r")
  data_list <- read_data(model=model, sim_type=sim_type, alpha=alpha, n=n, p=p, q=q, d=d, seed=i)
  X <- data_list$X
  Y <- data_list$Y
  Z <- data_list$Z
  
  h = 1.2*1.06*1*(4/(3*n))^{1/(1+4)}
  
  # pval1 corresponds to the proposed CIT of our paper
  test.stat =  CI.multiXYZ.test(X, Y, Z, h)
  pval1[i] = mean(boots.stat>test.stat)
  
  # pval3 corresponds to CDC 
  hdc = 1*1.06*sd(Z[,1])*(4/(3*n))^{1/(1+4)}
  pval2[i] = pdcor.test(X, Y, Z, R = 100)$p.value
  pval3[i] = cdcov.test(X, Y, Z,width = hdc)$p.value
  
  # pval4 corresponds to CMI
  cmi = cmiScores(X, Y, Z)
  boot.vec = rep(0, B)
  for (jj in 1:B) {
    #cat(jj, "\r")
    Xnew = local.boots.index(Z, X)
    boot.vec[jj] = cmiScores(Xnew, Y, Z)
  }
  pval4[i] = mean(boot.vec>cmi)
  
  # pval5 corresponds to KCI
  pval5[i] = KCI(X, Y, Z)$pvalue
}

### Report size and power
print("alpha is 0.05")
mean(pval1<0.05)
mean(pval2<0.05)
mean(pval3<0.05)
mean(pval4<0.05)
mean(pval5<0.05)

print("alpha is 0.1")
mean(pval1<0.1)
mean(pval2<0.1)
mean(pval3<0.1)
mean(pval4<0.1)
mean(pval5<0.1)

a1 = c(mean(pval1<0.05), mean(pval2<0.05), mean(pval3<0.05), mean(pval4<0.05), mean(pval5<0.05))
a2 = c(mean(pval1<0.1), mean(pval2<0.1), mean(pval3<0.1), mean(pval4<0.1), mean(pval5<0.1))

aa = rbind(a1, a2)
save(aa, file = rdaname)

# print elapsed time
new <- Sys.time() - old # calculate difference
print(new) # print in nice format
