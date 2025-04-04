# First: generate data
python model_generate_data.py --model=4 --sim_type=2 --alpha=.00 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.10 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.20 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.30 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3
python model_generate_data.py --model=4 --sim_type=2 --alpha=.40 --nsim=200 --d=50 --p=5 --q=5 --n=1000 --s=3


# trial code
# # python to execute our proposed test, CDC test, FCIT test, and CCIT test
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=0 --CCIT=1 --hidden_num=8 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-220 --nsim=200 &> model4_s2_a00.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=0 --CCIT=1 --hidden_num=8 --alpha=.10 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=1 --cpu=000-220 --nsim=200 &> model4_s2_a10.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=0 --CCIT=1 --hidden_num=8 --alpha=.20 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=2 --cpu=000-220 --nsim=200 &> model4_s2_a20.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=0 --CCIT=1 --hidden_num=8 --alpha=.30 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=3 --cpu=000-220 --nsim=200 &> model4_s2_a30.txt &
nohup python -u model.py --model=4 --sim_type=2 --FlowCIT=1 --FCIT=1 --CDC=0 --CCIT=1 --hidden_num=8 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=5 --gpu=4 --cpu=000-220 --nsim=200 &> model4_s2_a40.txt &

python model_result.py --model=4 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --hidden_num=8

# # # conda activate py37
# # # python code to execute GCIT
# nohup python -u model_GCIT.py --model=4 --sim_type=2 --alpha=.00 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=200-220 --nsim=200 --test_prop=.1 &> model4_s2_GCIT_a00.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=2 --alpha=.10 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.01 &> model4_s2_GCIT_a10.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=2 --alpha=.20 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.01 &> model4_s2_GCIT_a20.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=2 --alpha=.30 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.01 &> model4_s2_GCIT_a30.txt &
# nohup python -u model_GCIT.py --model=4 --sim_type=2 --alpha=.40 --n=1000 --p=5 --q=5 --d=50 --par_task=50 --cpu=200-220 --nsim=200 --test_prop=.1 &> model4_s2_GCIT_a40.txt &

# python model_result.py --model=4 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=1000 --p=5 --q=5 --d=50 --GCIT=True --GCIT_test_prop=.01

# Generate plots
python model_plot.py --model=4 --sim_type=2 --n=1000 --alphas="0.0,0.1,0.2,0.3,0.4" --legend=0 --KCI=0 --CLZ=0