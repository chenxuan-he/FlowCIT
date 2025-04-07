# 1. Generate data
python model_generate_data.py --model=4 --sim_type=2 --alpha=.00 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.10 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.20 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.30 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.40 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3

# 2. Python code to execute FlowCIT, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --hidden_num=8 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-220 --nsim=200 &> model4_s2_a00.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --hidden_num=8 --alpha=.10 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-220 --nsim=200 &> model4_s2_a10.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --hidden_num=8 --alpha=.20 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=000-220 --nsim=200 &> model4_s2_a20.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --hidden_num=8 --alpha=.30 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=3 --cpu=000-220 --nsim=200 &> model4_s2_a30.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --hidden_num=8 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=4 --cpu=000-220 --nsim=200 &> model4_s2_a40.txt &
python model_result.py --model=4 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --hidden_num=8

# 3. Python code to execute FlowCIT-IPC
nohup python -u model.py --model=4 --sim_type=2 --alpha=.0 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=0 --cpu=000-222 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=16 &> model4_s2_a00.txt &
nohup python -u model.py --model=4 --sim_type=2 --alpha=.1 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-222 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=16 &> model4_s2_a10.txt &
nohup python -u model.py --model=4 --sim_type=2 --alpha=.2 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=000-222 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=16 &> model4_s2_a20.txt &
nohup python -u model.py --model=4 --sim_type=2 --alpha=.3 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=3 --cpu=000-222 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=16 &> model4_s2_a30.txt &
nohup python -u model.py --model=4 --sim_type=2 --alpha=.4 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=3 --cpu=000-222 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=16 &> model4_s2_a40.txt &
python model_result.py --model=4 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --hidden_num=16

# 4. Generate plots
python model_plot.py --model=4 --sim_type=2 --n=1000 --alphas="0.0,0.1,0.2,0.3,0.4" --legend=0 --KCI=0 --CLZ=0