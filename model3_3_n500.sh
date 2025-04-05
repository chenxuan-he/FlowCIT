# 1. Generate data
python model_generate_data.py --model=3 --sim_type=3 --alpha=.00 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=3 --alpha=.10 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=3 --alpha=.20 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=3 --alpha=.30 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=3 --alpha=.40 --nsim=200 --d=2 --p=1 --q=1 --n=500

# 2. Use Rscript to execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=3 --sim_type=3 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s3_a00.txt 
nohup Rscript model_CLZ.R --model=3 --sim_type=3 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s3_a10.txt 
nohup Rscript model_CLZ.R --model=3 --sim_type=3 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s3_a20.txt 
nohup Rscript model_CLZ.R --model=3 --sim_type=3 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s3_a30.txt 
nohup Rscript model_CLZ.R --model=3 --sim_type=3 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s3_a40.txt 

# 3. Get result for CLZ and KCI
Rscript model_CLZ_result.R --model=3 --sim_type=3 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.00
Rscript model_CLZ_result.R --model=3 --sim_type=3 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.10
Rscript model_CLZ_result.R --model=3 --sim_type=3 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.20
Rscript model_CLZ_result.R --model=3 --sim_type=3 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.30
Rscript model_CLZ_result.R --model=3 --sim_type=3 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.40

# 4. Python code to execute FlowCIT, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=3 --sim_type=3 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=3 --cpu=000-040 --nsim=200 --hidden_num=32 &> model3_s3_a00.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=4 --cpu=040-080 --nsim=200 --hidden_num=32 &> model3_s3_a10.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=5 --cpu=080-120 --nsim=200 --hidden_num=32 &> model3_s3_a20.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=6 --cpu=120-160 --nsim=200 --hidden_num=32 &> model3_s3_a30.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=7 --cpu=160-200 --nsim=200 --hidden_num=32 &> model3_s3_a40.txt &
python model_result.py --model=3 --sim_type=3 --alphas="0.0,0.1,0.2,0.3,0.4" --n=500 --p=1 --q=1 --d=2 --hidden_num=32

# 5. Python code to execute FlowCIT-IPC
nohup python -u model.py --model=3 --sim_type=3 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=0 --cpu=100-200 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model3_s3_a00.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=1 --cpu=100-200 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model3_s3_a10.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=2 --cpu=100-200 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model3_s3_a20.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=6 --cpu=100-200 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model3_s3_a30.txt &
nohup python -u model.py --model=3 --sim_type=3 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=7 --cpu=100-200 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model3_s3_a40.txt &
python model_result.py --model=3 --sim_type=3 --alphas="0.0,0.1,0.2,0.3,0.4" --n=500 --p=1 --q=1 --d=2

# 6. Generate plots
python model_plot.py --model=3 --sim_type=3 --n=500 --alphas="0.0,0.1,0.2,0.3,0.4" --legend=0