# 1. Generate data
python model_generate_data.py --model=2 --sim_type=3 --alpha=.00 --nsim=200 --d=100 --p=50 --q=50 --n=1000 --s=2
python model_generate_data.py --model=2 --sim_type=3 --alpha=.20 --nsim=200 --d=100 --p=50 --q=50 --n=1000 --s=2
python model_generate_data.py --model=2 --sim_type=3 --alpha=.40 --nsim=200 --d=100 --p=50 --q=50 --n=1000 --s=2
python model_generate_data.py --model=2 --sim_type=3 --alpha=.60 --nsim=200 --d=100 --p=50 --q=50 --n=1000 --s=2
python model_generate_data.py --model=2 --sim_type=3 --alpha=.80 --nsim=200 --d=100 --p=50 --q=50 --n=1000 --s=2

# 2. Python code to execute FlowCIT, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=2 --sim_type=3 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=1 --cpu=000-210 --nsim=200 --FlowCIT=1 --CDC=1 --FCIT=1 --CCIT=1 --hidden_num=16 &> model2_s3_a00.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 --FlowCIT=1 --CDC=1 --FCIT=1 --CCIT=1 --hidden_num=16 &> model2_s3_a20.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=080-120 --nsim=200 --FlowCIT=1 --CDC=1 --FCIT=1 --CCIT=1 --hidden_num=16 &> model2_s3_a40.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=6 --cpu=120-160 --nsim=200 --FlowCIT=1 --CDC=1 --FCIT=1 --CCIT=1 --hidden_num=16 &> model2_s3_a60.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=160-200 --nsim=200 --FlowCIT=1 --CDC=1 --FCIT=1 --CCIT=1 --hidden_num=16 &> model2_s3_a80.txt &
python model_result.py --model=2 --sim_type=3 --alphas="0.0,0.2,0.4,0.6,0.8" --n=1000 --p=50 --q=50 --d=100 --hidden_num=16

# 3. Python code to execute FlowCIT-IPC
nohup python -u model.py --model=2 --sim_type=3 --alpha=.0 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=4 --cpu=000-080 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s3_a00.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.2 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=000-080 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s3_a20.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.4 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s3_a40.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.6 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s3_a60.txt &
nohup python -u model.py --model=2 --sim_type=3 --alpha=.8 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=100-220 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 --hidden_num=10 &> model2_s3_a80.txt &
python model_result.py --model=2 --sim_type=3 --alphas="0.0,0.2,0.4,0.6,0.8" --n=1000 --p=50 --q=50 --d=100 --hidden_num=10

# 4. Generate plots
python model_plot.py --model=2 --sim_type=3 --n=1000 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --CLZ=0 --KCI=0 --alphas="0.0,0.2,0.4,0.6,0.8" --legend=0