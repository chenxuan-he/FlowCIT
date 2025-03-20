# Simulation type 1: linear case
# # Rscript execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.0 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a0.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.1 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a1.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.2 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a2.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.3 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a3.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.4 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a4.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.5 --n=500 --p=3 --q=3 --d=3 --n_cpu=42 --bandwidth=.1 &> model1_CLZ_s1_a5.txt &

# # python execute our proposed test, CDC test, FCIT test, DGCIT
# # conda activate py37
# # the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# # model 1: (d_x, d_y, d_z, n)=(3, 3, 3, 500)
nohup python -u model.py --model=1 --sim_type=1 --alpha=.0 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=0 --nsim=200 &> model1_s1_a0.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.1 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --nsim=200 &> model1_s1_a1.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.2 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --nsim=200 &> model1_s1_a2.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.3 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --nsim=200 &> model1_s1_a3.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.4 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --nsim=200 &> model1_s1_a4.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.5 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=5 --nsim=200 &> model1_s1_a5.txt &

