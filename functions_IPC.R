IPC <- function(x, y) {
  n <- nrow(y)
  
  xxup <- x %*% t(x) + sum(colMeans(x^2))
  xxdown <- sqrt(diag(xxup))
  xxdown <- outer(xxdown, xxdown)
  b0 <- asin(xxup / xxdown)
  b <- Re(b0)
  diag(b) <- 0
  bjsum <- rowSums(b)
  bbsum <- sum(bjsum)
  
  yyup <- y %*% t(y) + sum(colMeans(y^2))
  yydown <- sqrt(diag(yyup))
  yydown <- outer(yydown, yydown)
  rhok0 <- asin(yyup / yydown)
  rhok <- Re(rhok0)
  diag(rhok) <- 0
  rhoksum <- rowSums(rhok)
  rhosum <- sum(rhoksum)
  
  S1 <- mean(rowSums(b * rhok)) / (n - 3)
  S2 <- mean(bjsum * rhoksum) / ((n - 3) * (n - 2))
  S3 <- rhosum * bbsum / (n * (n - 3) * (n - 2) * (n - 1))
  CVM2 <- S1 - 2 * S2 + S3
  
  S1 <- mean(rowSums(b * b)) / (n - 3)
  S2 <- mean(bjsum * bjsum) / ((n - 3) * (n - 2))
  S3 <- bbsum * bbsum / (n * (n - 3) * (n - 2) * (n - 1))
  vx <- S1 - 2 * S2 + S3
  
  S1 <- mean(rowSums(rhok * rhok)) / (n - 3)
  S2 <- mean(rhoksum * rhoksum) / ((n - 3) * (n - 2))
  S3 <- rhosum * rhosum / (n * (n - 3) * (n - 2) * (n - 1))
  vy <- S1 - 2 * S2 + S3
  
  IPC2 <- CVM2 / sqrt(vx * vy)
  ustat <- n * CVM2 / sqrt(2 * vx * vy)
  
  return(list(ustat = ustat, IPC2 = IPC2))
}
