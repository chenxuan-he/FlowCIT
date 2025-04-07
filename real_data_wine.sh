# baseline: X \indep Y \mid X
python real_data_wine.py --method="full"
Rscript real_data_wine.R --postfix="full"

# 1. PCA
python real_data_wine.py --method="pca" --latent_dim=2
Rscript real_data_wine.R --postfix="pca_2"

python real_data_wine.py --method="pca" --latent_dim=4
Rscript real_data_wine.R --postfix="pca_4"

python real_data_wine.py --method="pca" --latent_dim=11
Rscript real_data_wine.R --postfix="pca_11"

# 2. SIR
# conda activate py37: it requires numpy lower than 1.20
python real_data_wine.py --method="sir" --latent_dim=2
Rscript real_data_wine.R --postfix="sir_2"

python real_data_wine.py --method="sir" --latent_dim=4
Rscript real_data_wine.R --postfix="sir_4"

python real_data_wine.py --method="sir" --latent_dim=11
Rscript real_data_wine.R --postfix="sir_11"

# 3. UMAP
python real_data_wine.py --method="umap" --latent_dim=2
Rscript real_data_wine.R --postfix="umap_2"

python real_data_wine.py --method="umap" --latent_dim=4
Rscript real_data_wine.R --postfix="umap_4"

python real_data_wine.py --method="umap" --latent_dim=11
Rscript real_data_wine.R --postfix="umap_11"

# 4. DDR
# git clone https://github.com/Liao-Xu/DDR.git
# conda create --name ddr python=3.8
# conda activate ddr
# pip install -r requirements.txt
# I have keep one file in the folder ddr/model_reg.py which can be executed directly.
python real_data_wine_ddr.py --gpu="" --latent_dim=2
python real_data_wine.py --method="ddr" --latent_dim=2
Rscript real_data_wine.R --postfix="ddr_2"

