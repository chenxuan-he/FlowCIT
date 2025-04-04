# First: generate data
python model_generate_data.py --model=2 --sim_type=4 --alpha=.00 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=4 --alpha=.20 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=4 --alpha=.40 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=4 --alpha=.60 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=4 --alpha=.80 --nsim=200 --d=100 --p=50 --q=50 --n=1000

# # python to execute our proposed test, CDC test, FCIT test
nohup python -u model.py --model=2 --sim_type=4 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=0 --cpu=000-220 --nsim=200 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_s4_a00.txt & 
nohup python -u model.py --model=2 --sim_type=4 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=4 --cpu=000-050 --nsim=200 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_s4_a20.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=050-100 --nsim=200 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_s4_a40.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=6 --cpu=100-150 --nsim=200 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_s4_a60.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=150-200 --nsim=200 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_s4_a80.txt &

nohup python -u model.py --model=2 --sim_type=4 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=0 --cpu=000-220 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=1 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_FCIT_s4_a00.txt & 
nohup python -u model.py --model=2 --sim_type=4 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=4 --cpu=000-050 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=1 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_FCIT_s4_a20.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=050-100 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=1 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_FCIT_s4_a40.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=6 --cpu=100-150 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=1 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_FCIT_s4_a60.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=150-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=1 --CCIT=0 --hidden_num=6 --n_iter=300 &> model2_FCIT_s4_a80.txt &

nohup python -u model.py --model=2 --sim_type=4 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu="" --cpu=000-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --hidden_num=6 --n_iter=300 &> model2_s4_a00.txt & 
nohup python -u model.py --model=2 --sim_type=4 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu="" --cpu=000-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --hidden_num=6 --n_iter=300 &> model2_s4_a20.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu="" --cpu=000-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --hidden_num=6 --n_iter=300 &> model2_s4_a40.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu="" --cpu=000-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --hidden_num=6 --n_iter=300 &> model2_s4_a60.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu="" --cpu=000-200 --nsim=200 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --hidden_num=6 --n_iter=300 &> model2_s4_a80.txt &


python model_result.py --model=2 --sim_type=4 --alphas="0.0,0.2,0.4,0.6,0.8" --n=1000 --p=50 --q=50 --d=100 --hidden_num=6

# # add the FlowCIT-IPC
nohup python -u model.py --model=2 --sim_type=4 --alpha=.0 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=0 --cpu=000-080 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s4_a00.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.2 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=1 --cpu=000-080 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s4_a20.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.4 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s4_a40.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.6 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s4_a60.txt &
nohup python -u model.py --model=2 --sim_type=4 --alpha=.8 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s4_a80.txt &


# Generate plots
python model_plot.py --model=2 --sim_type=4 --n=1000 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --CLZ=0 --KCI=0 --alphas="0.0,0.2,0.4,0.6,0.8" --legend=0