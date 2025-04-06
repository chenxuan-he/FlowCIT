# baseline: X \indep Y \mid X
python real_data_wine.py --full=1
Rscript real_data_wine.R --postfix="full"

# 1. PCA
python real_data_wine.py --pca_2=1
Rscript real_data_wine.R --postfix="pca_2"

python real_data_wine.py --pca_4=1
Rscript real_data_wine.R --postfix="pca_4"

python real_data_wine.py --pca_full=1
Rscript real_data_wine.R --postfix="pca_full"

# 2. SIR
# conda activate py37: it requires numpy lower than 1.20
python real_data_wine.py --sir_2=1
Rscript real_data_wine.R --postfix="sir_2"

python real_data_wine.py --sir_4=1
Rscript real_data_wine.R --postfix="sir_4"

python real_data_wine.py --sir_full=1
Rscript real_data_wine.R --postfix="sir_full"

# 3. UMAP
python real_data_wine.py --umap_2=1
Rscript real_data_wine.R --postfix="umap_2"
