# read csv from file
read_data <- function(model, sim_type, alpha, n, p, q, d, seed){
    df <- read.csv(paste0("data/data_model", model, "_simtype", sim_type, "_alpha", sprintf("%.1f", alpha), "_n", n, "_p", p, "_q", q, "_d", d, "_seed", seed, ".csv"))
    # Extract X, Y, Z
    X <- as.matrix(df[, paste0("X", 1:p)])
    Y <- as.matrix(df[, paste0("Y", 1:q)])
    Z <- as.matrix(df[, paste0("Z", 1:d)])
    return(list(X=X, Y=Y, Z=Z))
}

