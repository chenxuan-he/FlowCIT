rm(list = ls())
set.seed(1234)
library(energy)
library(bnlearn)
library(Rcpp)
library(cdcsis)
library(CondIndTests)
library(praznik)

source("MI_test_functions.R")

#### Compare the proposed test with CDC, KCI, SW
old <- Sys.time() # get start time


### Example 1: X = (X1, X2), Y = (Y1, Y2), Z = (Z1, Z2)
############# Y1 = m*X1+Z1

n = 50
m = 0
simutime = 1000
B = 100
rdaname = paste("multiXYZ", "-", "1", "-", n, ".rda", sep = "")



# get the null distribution (generate x y z with the same sample size and dimensions as the original data)
filename = paste("multi_m1.",n,".rda", sep = "")
if(file.exists(filename)) {load(filename);print("Already have bootstrap file!")} else {
  #### Bootstrap the null distribution
  print("Now start bootstrap!")
  h = 1.2*1.06*1*(4/(3*n))^{1/(1+4)}
  boots.time = 1000
  boots.stat = rep(0, boots.time)
  for (i in 1:boots.time) {
    if(i%%100==0) {print(i)}
    X = matrix(rnorm(n*2), n, 2)
    Y = matrix(rnorm(n*2), n, 2)
    Z = matrix(rnorm(n*2), n, 2)
    boots.stat[i] = CI.multiXYZ.test(X, Y, Z, h)
  }
  save(boots.stat, file = filename)
  print("Finished bootstrap!")
}


#### Independence Test
pval1 <- pval2 <- pval3 <- pval4 <- pval5 <- rep(0, simutime)

for (i in 1:simutime) {
  cat(i, "\r")
  X = matrix(rnorm(n*2), n, 2)
  Y = matrix(rnorm(n*2), n, 2)
  Z = matrix(rnorm(n*2), n, 2)
  X[,1] = X[,1] + Z[,1]
  Y[,1] = m*X[,1] + Z[,1] + Z[,2]
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


