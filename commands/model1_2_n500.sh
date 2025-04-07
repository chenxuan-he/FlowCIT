# 1. Generate data
python model_generate_data.py --model=1 --sim_type=2 --alpha=.00 --nsim=200 --d=3 --p=3 --q=3 --n=500
python model_generate_data.py --model=1 --sim_type=2 --alpha=.05 --nsim=200 --d=3 --p=3 --q=3 --n=500
python model_generate_data.py --model=1 --sim_type=2 --alpha=.10 --nsim=200 --d=3 --p=3 --q=3 --n=500
python model_generate_data.py --model=1 --sim_type=2 --alpha=.15 --nsim=200 --d=3 --p=3 --q=3 --n=500
python model_generate_data.py --model=1 --sim_type=2 --alpha=.20 --nsim=200 --d=3 --p=3 --q=3 --n=500

# 2. Use Rscript to execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=1 --sim_type=2 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s2_a00.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=2 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s2_a05.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=2 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s2_a10.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=2 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s2_a15.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=2 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s2_a20.txt &

# 3. Get result for CLZ and KCI
Rscript model_CLZ_result.R --model=1 --sim_type=2 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.00
Rscript model_CLZ_result.R --model=1 --sim_type=2 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.05
Rscript model_CLZ_result.R --model=1 --sim_type=2 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.10
Rscript model_CLZ_result.R --model=1 --sim_type=2 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.15
Rscript model_CLZ_result.R --model=1 --sim_type=2 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.20

# 4. Python code to execute FlowCIT, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=1 --sim_type=2 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 --hidden_num=32 &> model1_s2_a00.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 --hidden_num=32 &> model1_s2_a05.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 --hidden_num=32 &> model1_s2_a10.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 --hidden_num=32 &> model1_s2_a15.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 --hidden_num=32 &> model1_s2_a20.txt &
python model_result.py --model=1 --sim_type=2 --alphas="0.0,0.05,0.1,0.15,0.2" --n=500 --p=3 --q=3 --d=3 --hidden_num=32 

# 5. Python code to execute FlowCIT-IPC
nohup python -u model.py --model=1 --sim_type=2 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=0 --cpu=200-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=20 &> model1_s2_a00.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=200-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=20 &> model1_s2_a05.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=200-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=20 &> model1_s2_a10.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=200-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=20 &> model1_s2_a15.txt &
nohup python -u model.py --model=1 --sim_type=2 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=200-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=20 &> model1_s2_a20.txt &
python model_result.py --model=1 --sim_type=2 --alphas="0.0,0.05,0.1,0.15,0.2" --n=500 --p=3 --q=3 --d=3 --hidden_num=20

# 6. Generate plots
python model_plot.py --model=1 --sim_type=2 --n=500 --alphas="0.0,0.05,0.1,0.15,0.2" --legend=0