# First: generate data
python model_generate_data.py --model=2 --sim_type=1 --alpha=.00 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=1 --alpha=.20 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=1 --alpha=.40 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=1 --alpha=.60 --nsim=200 --d=100 --p=50 --q=50 --n=1000
python model_generate_data.py --model=2 --sim_type=1 --alpha=.80 --nsim=200 --d=100 --p=50 --q=50 --n=1000

# # python to execute our proposed test, CDC test, FCIT test
nohup python -u model.py --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=10 --gpu=3 --cpu=000-215 --nsim=200 --hidden_num=24 --n_iter=800 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 --lr=5e-3 &> model2_s1_a00_neuron24_iter800.txt &

# hidden_num 8, n_iter 1000, lr=1e-3
# hidden_num 16, n_iter 400, 0.19
# hidden_num 16, n_iter 800, 0.095

# hidden_num 20, n_iter 500, 0.115
# hidden_num 20, n_iter 800, 0.08
# hidden_num 20, n_iter 1000, 0.08
# hidden_num 20, n_iter 1000, lr=1e-3, xxx

# hidden_num 24, n_iter 800, 

# hidden_num 25, n_iter 800, 0.105

# hidden_num 32, n_iter 400, 0.11
# hidden_num 32, n_iter 600, 0.105
# hidden_num 32, n_iter 800, 0.08
# hidden_num 32, n_iter 1000, 0.105

# hidden_num 40, n_iter 800, 0.13

# hidden_num 64, n_iter 500, 0.16


nohup python -u model.py --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=4 --cpu=040-080 --nsim=200 --hidden_num=64 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 &> model2_s1_a20.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=080-120 --nsim=200 --hidden_num=64 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 &> model2_s1_a40.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=6 --cpu=120-160 --nsim=200 --hidden_num=64 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 &> model2_s1_a60.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=7 --cpu=160-200 --nsim=200 --hidden_num=64 --FlowCIT=1 --CDC=0 --FCIT=0 --CCIT=0 &> model2_s1_a80.txt &

python model_result.py --model=2 --sim_type=1 --alphas="0.0" --n=1000 --p=50 --q=50 --d=100 --hidden_num=32

# # python: add CCIT test
# Running on the 15 machine
nohup python -u model.py --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --cpu=040-255 --nsim=200 &> model2_CCIT_s1_a00.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --cpu=040-255 --nsim=200 &> model2_CCIT_s1_a20.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --cpu=040-255 --nsim=200 &> model2_CCIT_s1_a40.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --cpu=040-255 --nsim=200 &> model2_CCIT_s1_a60.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --FlowCIT=0 --CDC=0 --FCIT=0 --CCIT=1 --cpu=040-255 --nsim=200 &> model2_CCIT_s1_a80.txt &

python model_result.py --model=2 --sim_type=1 --alphas="0.0,0.2,0.4,0.6,0.8" --n=1000 --p=50 --q=50 --d=100


# # # conda activate py37
# # # python code to execute GCIT
# nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=000-040 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a00.txt &
# nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a20.txt &
# nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.40 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a40.txt &
# nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.60 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a60.txt &
# nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.80 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=160-200 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a80.txt &

# python model_result.py --model=2 --sim_type=1 --alphas="0.0,0.2,0.4,0.6,0.8" --n=1000 --p=50 --q=50 --d=100 --GCIT=True --GCIT_test_prop=.02

# Generate plots
python model_plot.py --model=2 --sim_type=1 --n=1000 --FlowCIT=1 --FCIT=1 --CDC=1 --CCIT=1 --CLZ=0 --KCI=0 --alphas="0.0,0.2,0.4,0.6,0.8" --legend=1