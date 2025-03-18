library(bnlearn)
library(Rcpp)
library(cdcsis)
choose.h.size.CI = function(simutime, hrange, boots.stat, consider0.1 = TRUE)
{
  l = length(hrange)
  s0.05 <-  s0.1 <- cri <- rep(0, l)
  for(jj in 1:l)
  {
    print(jj)
    h = hrange[jj]*1.06*1*(4/(3*n))^{1/(1+4)}
    #### Independence Test
    pval1 = rep(0, simutime)
    for (i in 1:simutime) {
      #### Build the conditional model
      X1 = rnorm(n)
      X2 = rnorm(n)
      Z = rnorm(n)
      X = X1+Z
      Y = X2+Z
      test.stat =  CI.test(X, Y, Z, h)
      pval1[i] = mean(boots.stat>test.stat)
    }
    ### Report size
    s0.05[jj] = mean(pval1<0.05)
    s0.1[jj] = mean(pval1<0.1)
    if(consider0.1){ cri[jj] = abs(s0.05[jj]-0.05)+abs(s0.1[jj]-0.1)} else { cri[jj] = abs(s0.05[jj]-0.05) }
  }
  minh = which.min(cri)
  return(list(h = hrange[minh]*1.06*1*(4/(3*n))^{1/(1+4)}, multiplier = hrange[minh], size0.05 = s0.05[minh], size0.1 = s0.1[minh]))
}

choose.h.size.CDC = function(simutime, hrange, consider0.1 = TRUE)
{
  l = length(hrange)
  s0.05 <-  s0.1 <- cri <- rep(0, l)
  for(jj in 1:l)
  {
    print(jj)
    h = hrange[jj]*1.06*1*(4/(3*n))^{1/(1+4)}
    #### Independence Test
    pval1 = rep(0, simutime)
    for (i in 1:simutime) {
      if(i%%(simutime/10)==0) {print(i)}
      #### Build the conditional model
      X1 = rnorm(n)
      X2 = rnorm(n)
      Z = rnorm(n)
      X = X1+Z
      Y = X2+Z
      #### Transform into matrices
      X = as.matrix(X)
      Y = as.matrix(Y)
      Z = as.matrix(Z)
      pval1[i] = cdcov.test(X, Y, Z, width = h)$p.value
    }
    ### Report size
    s0.05[jj] = mean(pval1<0.05)
    s0.1[jj] = mean(pval1<0.1)
    if(consider0.1){ cri[jj] = abs(s0.05[jj]-0.05)+abs(s0.1[jj]-0.1)} else { cri[jj] = abs(s0.05[jj]-0.05) }
  }
  minh = which.min(cri)
  return(list(h = hrange[minh]*1.06*1*(4/(3*n))^{1/(1+4)}, multiplier = hrange[minh], size0.05 = s0.05[minh], size0.1 = s0.1[minh]))
}


DAG.random = function(p, s, sigma)
{
  A = matrix(0, p, p)
  for(i in 1:p)
  {
    for(j in 1:p)
    {
      if(i>j) 
      { tmp = rbinom(1,1,s)
      if(tmp) {A[i,j] = runif(1, min = 0.1, max = 1)}
      }
    }
  }
  X = matrix(0, n, p)
  X[,1] = rnorm(n)
  for(i in 2:p)
  {
    X[,i] = X%*%A[i,]+sigma*rnorm(n)
  }
  list(X=X,A=A)
}
CI.pc.test = function(x, y, S, suffStat) 
{
  B = suffStat$B
  X.design = suffStat$X
  h = suffStat$h
  kk = length(S)
  #print(kk)
  if(kk==0)
  {
    #print(kk)
    t1 = InDe.test(X.design[,x],X.design[,y])
    return(mean(B[kk+1,]>t1))
  }
  if(kk!=0)
  {
    #print(kk)
    t1 = CI.test(X.design[,x],X.design[,y],X.design[,S],h)
    return(mean(B[kk+1,]>t1))
  }
}

mutualDependenceFree = function(u,v,w)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else { p = dim(u)[2] }
  if(is.null(dim(v))) {q = 1} else { q = dim(u)[2] }
  if(is.null(dim(w))) {r = 1} else { r = dim(u)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  ww = as.matrix(exp(-dist(w, diag = T, upper = T)))
  conCor = mean(uu*vv*ww)
  conCor
}

mutualDependenceFree_multiZ = function(u,v,w)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else {stop("X should be univariate!") }
  if(is.null(dim(v))) {q = 1} else {stop("Y should be univariate!") }
  if(is.null(dim(w))) {stop("Z should be multivariate!")} else { r = dim(w)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  ww = as.matrix(exp(-wdist(w)))
  conCor = mean(uu*vv*ww)
  conCor
}

mutualDependenceFree_multiXYZ = function(u,v,w)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else { p = dim(u)[2] }
  if(is.null(dim(v))) {q = 1} else { q = dim(v)[2] }
  if(is.null(dim(w))) {r = 1} else { r = dim(w)[2] }
  
  conCor = 0
  for (k in 1:(n-1)) {
    for(l in (k+1):n){
      conCor = conCor + multiUV(u[k,], u[l,])*multiUV(v[k,], v[l,])*exp(-sum(abs(w[k,]-w[l,])))
    }
  }
  conCor/(n*(n-1)/2)
}

### Uk and Ul are two sample vectors from U or V
multiUV = function(Uk, Ul)
{
  p = length(Uk)
  exp(-sum(abs(Uk-Ul))) + (2/exp(1))^p - prod(-exp(Uk-1)-exp(-Uk)+2) - prod(-exp(Ul-1)-exp(-Ul)+2)
}

wdist = function(w)
{
  if(is.null(dim(w))) {stop("Z should be multivariate!")} else { r = dim(w)[2] }
  s1 = dist(w[,1], diag = T, upper = T)
  for(i in 2:r)
  {
    s1 = s1+dist(w[,i], diag = T, upper = T)
  }
  s1
}

IndependenceFree = function(u,v)
{
  if(is.null(dim(u))) {n = length(u)} else { n = dim(u)[1] }
  if(is.null(dim(u))) {p = 1} else { p = dim(u)[2] }
  if(is.null(dim(v))) {q = 1} else { q = dim(u)[2] }
  #if(is.null(dim(w))) {r = 1} else { r = dim(u)[2] }
  uij = as.matrix(dist(u, diag = T, upper = T))
  ui = exp(-u)+exp(u-1)
  vij = as.matrix(dist(v, diag = T, upper = T))
  vi = exp(-v)+exp(v-1)
  uu = exp(-uij)+ui%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(ui)+2*exp(-1)-4
  vv = exp(-vij)+vi%*%matrix(1, 1, n)+matrix(1, n, 1)%*%t(vi)+2*exp(-1)-4
  #ww = as.matrix(exp(-dist(w, diag = T, upper = T)))
  conCor = mean(uu*vv)
  conCor
}

### Define kernel functions
k_gaussian = function(t, h)
{
  u = t/h
  return(exp(-t(u)%*%u/2)/sqrt(2*pi))
}
k_ep = function(t, h)
{
  u = t/h
  return(0.75*(1-u^2)*(u<=1)*(u>=-1))
}


UEstimate = function(x1, z1, X, Z, h)
{
  #if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  ZK = GetZK(z1, Z, h)
  uNum = mean(ZK*(X<=x1))
  return(uNum/mean(ZK))
}

VEstimate = function(y1, z1, Y, Z, h)
{
  ZK = GetZK(z1, Z, h)
  vNum = mean(ZK*(Y<=y1))
  return(vNum/mean(ZK))
}

WEstimate = function(z1, Z)
{
  return(mean(Z<=z1))
}


GetZK = function(z1, Z, h)
{
  if(is.null(nrow(Z))) 
  {
    n = length(Z)
    Kh = rep(0,n)
    for (i in 1:n) {
      Kh[i] = k_gaussian(z1-Z[i], h)
    }
    return(Kh)
  }
  if(!is.null(nrow(Z)))  {
    n = dim(Z)[1]
    Kh = rep(0,n)
    for (i in 1:n) {
      Kh[i] = k_gaussian(z1-Z[i,], h)
    }
    return(Kh)
  }
}

CI.test = function(X, Y, Z, h)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  if(is.null(dim(X))) {p = 1} else {stop("X should be univariate!") }
  if(is.null(dim(Y))) {q = 1} else {stop("Y should be univariate!") }
  if(is.null(dim(Z))) {r = 1} else { r = dim(Z)[2] }
  
  if(r==1)
  {
    ### Estimate U, V, W
    U = rep(0, n)
    V = U; W = U
    for (i in 1:n) {
      U[i] = UEstimate(X[i], Z[i], X, Z, h)
      V[i] = VEstimate(Y[i], Z[i], Y, Z, h)
      W[i] = WEstimate(Z[i], Z)
    }
    conCor = mutualDependenceFree(U,V,W)
    return(conCor)
  }
  if(r>1)
  {
    #### Estimate U, V, W
    U = rep(0, n)
    V = U; W = matrix(0, n, r)
    for (i in 1:n) {
      U[i] = UEstimate(X[i], Z[i], X, Z, h)
      V[i] = VEstimate(Y[i], Z[i], Y, Z, h)
      for(j in 1:r)
      {
        if(j == 1){ W[i,j] = WEstimate(Z[i,j], Z[,j]) } else { W[i,j] = UEstimate(Z[i,j], Z[i,1:(j-1)], Z[,j], Z[,1:(j-1)],h)}
      }
    }
    conCor = mutualDependenceFree_multiZ(U,V,W)
    return(conCor)
  }
}

#### When the conditioning Z is multivariate
CI.multiZ.test = function(X, Y, Z, h)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  if(is.null(dim(X))) {p = 1} else {stop("X should be univariate!") }
  if(is.null(dim(Y))) {q = 1} else {stop("Y should be univariate!") }
  if(is.null(dim(Z))) {stop("Z should be multivariate!")} else { r = dim(Z)[2] }
  #### Estimate U, V, W
  U = rep(0, n)
  V = U; W = matrix(0, n, r)
  for (i in 1:n) {
    U[i] = UEstimate(X[i], Z[i], X, Z, h)
    V[i] = VEstimate(Y[i], Z[i], Y, Z, h)
    for(j in 1:r)
    {
      if(j == 1){ W[i,j] = WEstimate(Z[i,j], Z[,j]) } else { W[i,j] = UEstimate(Z[i,j], Z[i,1:(j-1)], Z[,j], Z[,1:(j-1)],h)}
      
    }
  }
  conCor = mutualDependenceFree_multiZ(U,V,W)
  conCor
}

#### When the all X, Y, Z are multivariate
CI.multiXYZ.test = function(X, Y, Z, h)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1]}
  if(is.null(dim(X))) {p = 1} else { p = dim(X)[2] }
  if(is.null(dim(Y))) {q = 1} else { q = dim(Y)[2] }
  if(is.null(dim(Z))) {r = 1} else { r = dim(Z)[2] }
  #### Estimate U, V, W
  U = matrix(0, n, p)
  V = matrix(0, n, q)
  W = matrix(0, n, r)
  for (i in 1:n) {
    for(j in 1:p)
    {
      if(j == 1){ U[i,j] = UEstimate(X[i,j], Z[i,], X[,j], Z, h) } else { U[i,j] = UEstimate(X[i,j], c(X[i,1:(j-1)], Z[i,]), X[,j], cbind(X[,1:(j-1)], Z), h)}
    }
    for(j in 1:q)
    {
      if(j == 1){ V[i,j] = UEstimate(Y[i,j], Z[i,], Y[,j], Z, h) } else { V[i,j] = UEstimate(Y[i,j], c(Y[i,1:(j-1)], Z[i,]), Y[,j], cbind(Y[,1:(j-1)], Z), h)}
    }
    for(j in 1:r)
    {
      if(j == 1){ W[i,j] = WEstimate(Z[i,j], Z[,j]) } else { W[i,j] = UEstimate(Z[i,j], Z[i,1:(j-1)], Z[,j], Z[,1:(j-1)],h)}
      
    }
  }
  conCor = mutualDependenceFree_multiXYZ(U,V,W)
  conCor
}

InDe.test = function(X, Y)
{
  if(is.null(nrow(X))) {n = length(X)} else {n = dim(X)[1,]}
  if(is.null(dim(X))) {p = 1} else { p = dim(X)[2] }
  if(is.null(dim(Y))) {q = 1} else { q = dim(Y)[2] }
  #if(is.null(dim(Z))) {r = 1} else { r = dim(Z)[2] }
  
  #### Estimate U, V, 
  U = rep(0, n); V = U;
  for (i in 1:n) {
    U[i] = WEstimate(X[i], X)
    V[i] = WEstimate(Y[i], Y)
  }
  conCor = IndependenceFree(U,V)
  conCor
}



ConditionalCDF = function(X,Y,X0,Y0, h){
  n = nrow(X); p = ncol(X)
  m = length(Y)
  n0 = nrow(X0); p0 = ncol(X0)
  m0 = length(Y0)
  #if(p==1) { h = density(X)$bw } else { h = density(X[,1])$bw }
  KerUni = array(0, dim = c(n0,n,p))
  for(dim in 1:p){
    Xt = X[,dim]
    Xt0 = X0[,dim]
    U = (Xt0%*%matrix(1, 1, n) - matrix(1, n0, 1)%*%t(Xt))/h
    KerUni[,,dim] = exp(-0.5*U^2)/sqrt(2*pi)
  }
  KerWgt = apply(KerUni, c(1,2), prod)
  KerSum = rowSums(KerWgt)
  CDF = KerWgt %*%(Y%*% matrix(1, 1, m0) <= matrix(1, n, 1)%*%t(Y0)) / (KerSum%*%matrix(1, 1, m0))
  CDF
}

### The goal is to generate local bootstrap for Y, which is multivariate.
local.boots.index = function(X, Ymulti){
  Y = Ymulti[,1]
  n = length(Y)
  sortY = Y[order(Y)]
  sortYmulti = Ymulti[order(Y),]
  X0 = X
  h = 2*density(X[,1])$bw
  fhatnew = ConditionalCDF(X, Y, X0, sortY, h)
  r1 = matrix(rep(runif(n), n), n, n)
  ind1 = apply(abs(fhatnew - r1), 1, which.min)
  sortYmulti[ind1,]
}


