# First: generate data
python model_generate_data.py --model=4 --sim_type=1 --alpha=.00 --nsim=200 --d=50 --p=5 --q=5 --n=1000
python model_generate_data.py --model=4 --sim_type=1 --alpha=.10 --nsim=200 --d=50 --p=5 --q=5 --n=1000
python model_generate_data.py --model=4 --sim_type=1 --alpha=.20 --nsim=200 --d=50 --p=5 --q=5 --n=1000
python model_generate_data.py --model=4 --sim_type=1 --alpha=.30 --nsim=200 --d=50 --p=5 --q=5 --n=1000
python model_generate_data.py --model=4 --sim_type=1 --alpha=.40 --nsim=200 --d=50 --p=5 --q=5 --n=1000


# trial code
nohup python -u model.py --model=4 --sim_type=1 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=210-220 --nsim=20 --CCIT=0 &> model4_s1_a00.txt &
nohup python -u model.py --model=4 --sim_type=1 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=200-210 --nsim=20 --CCIT=0 &> model4_s1_a40.txt &


# # python to execute our proposed test, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=4 --sim_type=1 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 &> model4_s1_a00.txt &
nohup python -u model.py --model=4 --sim_type=1 --alpha=.10 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 &> model4_s1_a10.txt &
nohup python -u model.py --model=4 --sim_type=1 --alpha=.20 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 &> model4_s1_a20.txt &
nohup python -u model.py --model=4 --sim_type=1 --alpha=.30 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 &> model4_s1_a30.txt &
nohup python -u model.py --model=4 --sim_type=1 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 &> model4_s1_a40.txt &

python model_result.py --model=4 --sim_type=1 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --hidden_num=64 

# # # conda activate py37
# # # python code to execute GCIT
# nohup python -u model_GCIT.py --model=4 --sim_type=1 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=200-220 --nsim=200 --test_prop=.1 &> model4_s1_GCIT_a00.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=1 --alpha=.10 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.01 &> model4_s1_GCIT_a10.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=1 --alpha=.20 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.01 &> model4_s1_GCIT_a20.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=1 --alpha=.30 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.01 &> model4_s1_GCIT_a30.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=1 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=200-220 --nsim=200 --test_prop=.1 &> model4_s1_GCIT_a40.txt &

# python model_result.py --model=4 --sim_type=1 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --GCIT=True --GCIT_test_prop=.01

# Generate plots
python model_plot.py --model=4 --sim_type=1 --n=1000 --alphas="0.0,0.1,0.2,0.3,0.4" --legend=1