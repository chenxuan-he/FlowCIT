# Simulation type 1: linear case
# # Rscript execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --n_cpu=50 --bandwidth=.2 &> model1_CLZ_s1_a00.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --n_cpu=50 --bandwidth=.2 &> model1_CLZ_s1_a05.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --n_cpu=50 --bandwidth=.2 &> model1_CLZ_s1_a10.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --n_cpu=50 --bandwidth=.2 &> model1_CLZ_s1_a15.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=1 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --n_cpu=50 --bandwidth=.2 &> model1_CLZ_s1_a20.txt &

# get result for CLZ and KCI
Rscript model_CLZ_result.R --model=1 --sim_type=1 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.00
Rscript model_CLZ_result.R --model=1 --sim_type=1 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.05
Rscript model_CLZ_result.R --model=1 --sim_type=1 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.10
Rscript model_CLZ_result.R --model=1 --sim_type=1 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.15
Rscript model_CLZ_result.R --model=1 --sim_type=1 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.20

# # python execute our proposed test, CDC test, FCIT test, DGCIT
# # conda activate py37
# # the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# # model 1: (d_x, d_y, d_z, n)=(3, 3, 3, 500)
nohup python -u model.py --model=1 --sim_type=1 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=0 --cpu=000-040 --nsim=200 &> model1_s1_a00.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=040-080 --nsim=200 &> model1_s1_a05.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=080-120 --nsim=200 &> model1_s1_a10.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=120-160 --nsim=200 &> model1_s1_a15.txt &
nohup python -u model.py --model=1 --sim_type=1 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=160-200 --nsim=200 &> model1_s1_a20.txt &

