# read csv from file
read_data <- functions(model, seed, p, q, d){
    df <- read.csv(paste0("data/data_model", model, "_simtype", sim_type, "_alpha", alpha, "_n", n, "_p", p, "_q", q, "_d", d, "_seed", seed, ".csv"))
    # Extract X, Y, Z
    X <- as.matrix(df[, paste0("X", 1:p)])
    Y <- as.matrix(df[, paste0("X", 1:q)])
    Z <- as.matrix(df[, paste0("X", 1:d)])
    return(list(X=X, Y=Y, Z=Z))
}
