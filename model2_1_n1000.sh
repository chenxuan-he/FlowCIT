# First: generate data
nohup python model_generate_data.py --model=2 --sim_type=1 --alpha=.00 --nsim=200 --d=100 --p=50 --q=50 --n=1000
nohup python model_generate_data.py --model=2 --sim_type=1 --alpha=.05 --nsim=200 --d=100 --p=50 --q=50 --n=1000
nohup python model_generate_data.py --model=2 --sim_type=1 --alpha=.10 --nsim=200 --d=100 --p=50 --q=50 --n=1000
nohup python model_generate_data.py --model=2 --sim_type=1 --alpha=.15 --nsim=200 --d=100 --p=50 --q=50 --n=1000
nohup python model_generate_data.py --model=2 --sim_type=1 --alpha=.20 --nsim=200 --d=100 --p=50 --q=50 --n=1000

# # use Rscript to execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --n_cpu=40 --bandwidth=.2 &> model2_CLZ_s1_a00.txt &
nohup Rscript model_CLZ.R --model=2 --sim_type=1 --alpha=.05 --n=1000 --p=50 --q=50 --d=100 --n_cpu=40 --bandwidth=.2 &> model2_CLZ_s1_a05.txt &
nohup Rscript model_CLZ.R --model=2 --sim_type=1 --alpha=.10 --n=1000 --p=50 --q=50 --d=100 --n_cpu=40 --bandwidth=.2 &> model2_CLZ_s1_a10.txt &
nohup Rscript model_CLZ.R --model=2 --sim_type=1 --alpha=.15 --n=1000 --p=50 --q=50 --d=100 --n_cpu=40 --bandwidth=.2 &> model2_CLZ_s1_a15.txt &
nohup Rscript model_CLZ.R --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --n_cpu=40 --bandwidth=.2 &> model2_CLZ_s1_a20.txt &

# get result for CLZ and KCI
Rscript model_CLZ_result.R --model=2 --sim_type=1 --n=1000 --p=50 --q=50 --d=100 --bandwidth=.2 --alpha=.00
Rscript model_CLZ_result.R --model=2 --sim_type=1 --n=1000 --p=50 --q=50 --d=100 --bandwidth=.2 --alpha=.05
Rscript model_CLZ_result.R --model=2 --sim_type=1 --n=1000 --p=50 --q=50 --d=100 --bandwidth=.2 --alpha=.10
Rscript model_CLZ_result.R --model=2 --sim_type=1 --n=1000 --p=50 --q=50 --d=100 --bandwidth=.2 --alpha=.15
Rscript model_CLZ_result.R --model=2 --sim_type=1 --n=1000 --p=50 --q=50 --d=100 --bandwidth=.2 --alpha=.20

# # python to execute our proposed test, CDC test, FCIT test
nohup python -u model.py --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 &> model2_s1_a00.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.05 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 &> model2_s1_a05.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.10 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 &> model2_s1_a10.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.15 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 &> model2_s1_a15.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 &> model2_s1_a20.txt &

python model_result.py --model=2 --sim_type=1 --alphas="0.0,0.05,0.1,0.15,0.2" --n=1000 --p=50 --q=50 --d=100 --hidden_num=64 

# # conda activate py37
# # python code to execute GCIT
nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.00 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=000-040 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a00.txt &
nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.05 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a05.txt &
nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.10 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a10.txt &
nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.15 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a15.txt &
nohup python -u model_GCIT.py --model=2 --sim_type=1 --alpha=.20 --n=1000 --p=50 --q=50 --d=100 --par_task=50 --cpu=160-200 --nsim=200 --test_prop=.02 &> model2_s1_GCIT_a20.txt &

python model_result.py --model=2 --sim_type=1 --alphas="0.0,0.05,0.1,0.15,0.2" --n=1000 --p=50 --q=50 --d=100 --hidden_num=64 --GCIT=True --GCIT_test_prop=.02

# Generate plots
python model_plot.py --model=2 --sim_type=1 --n=1000 --alphas="0.0,0.05,0.1,0.15,0.2" --legend=0