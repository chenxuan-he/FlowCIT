# First: generate data
python model_generate_data.py --model=3 --sim_type=2 --alpha=.00 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=2 --alpha=.10 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=2 --alpha=.20 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=2 --alpha=.30 --nsim=200 --d=2 --p=1 --q=1 --n=500
python model_generate_data.py --model=3 --sim_type=2 --alpha=.40 --nsim=200 --d=2 --p=1 --q=1 --n=500

# # use Rscript to execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=3 --sim_type=2 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s2_a00.txt &
nohup Rscript model_CLZ.R --model=3 --sim_type=2 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s2_a10.txt &
nohup Rscript model_CLZ.R --model=3 --sim_type=2 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s2_a20.txt &
nohup Rscript model_CLZ.R --model=3 --sim_type=2 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s2_a30.txt &
nohup Rscript model_CLZ.R --model=3 --sim_type=2 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --n_cpu=40 --bandwidth=.5 &> model3_CLZ_s2_a40.txt &

# get result for CLZ and KCI
Rscript model_CLZ_result.R --model=3 --sim_type=2 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.00
Rscript model_CLZ_result.R --model=3 --sim_type=2 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.10
Rscript model_CLZ_result.R --model=3 --sim_type=2 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.20
Rscript model_CLZ_result.R --model=3 --sim_type=2 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.30
Rscript model_CLZ_result.R --model=3 --sim_type=2 --n=500 --p=1 --q=1 --d=2 --bandwidth=.5 --alpha=.40

# # python to execute our proposed test, CDC test, FCIT test
nohup python -u model.py --model=3 --sim_type=2 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 --hidden_num=32 &> model3_s2_a00.txt &
nohup python -u model.py --model=3 --sim_type=2 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 --hidden_num=32 &> model3_s2_a10.txt &
nohup python -u model.py --model=3 --sim_type=2 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 --hidden_num=32 &> model3_s2_a20.txt &
nohup python -u model.py --model=3 --sim_type=2 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 --hidden_num=32 &> model3_s2_a30.txt &
nohup python -u model.py --model=3 --sim_type=2 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 --hidden_num=32 &> model3_s2_a40.txt &

python model_result.py --model=3 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=500 --p=1 --q=1 --d=2 --hidden_num=32

# # conda activate py37
# # python code to execute GCIT
nohup python -u model_GCIT.py --model=3 --sim_type=2 --alpha=.00 --n=500 --p=1 --q=1 --d=2 --par_task=50 --cpu=000-040 --nsim=200 --test_prop=.01 &> model3_s2_GCIT_a00.txt &
nohup python -u model_GCIT.py --model=3 --sim_type=2 --alpha=.10 --n=500 --p=1 --q=1 --d=2 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.01 &> model3_s2_GCIT_a10.txt &
nohup python -u model_GCIT.py --model=3 --sim_type=2 --alpha=.20 --n=500 --p=1 --q=1 --d=2 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.01 &> model3_s2_GCIT_a20.txt &
nohup python -u model_GCIT.py --model=3 --sim_type=2 --alpha=.30 --n=500 --p=1 --q=1 --d=2 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.01 &> model3_s2_GCIT_a30.txt &
nohup python -u model_GCIT.py --model=3 --sim_type=2 --alpha=.40 --n=500 --p=1 --q=1 --d=2 --par_task=50 --cpu=160-200 --nsim=200 --test_prop=.01 &> model3_s2_GCIT_a40.txt &

python model_result.py --model=3 --sim_type=2 --alphas="0.0,0.1,0.2,0.3,0.4" --n=500 --p=1 --q=1 --d=2 --GCIT=True --GCIT_test_prop=.01

# Generate plots
python model_plot.py --model=3 --sim_type=2 --n=500 --alphas="0.0,0.1,0.2,0.3,0.4" --legend=0