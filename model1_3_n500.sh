# First: generate data
nohup python model_generate_data.py --model=1 --sim_type=3 --alpha=.00 --nsim=200 --d=3 --p=3 --q=3 --n=500
nohup python model_generate_data.py --model=1 --sim_type=3 --alpha=.05 --nsim=200 --d=3 --p=3 --q=3 --n=500
nohup python model_generate_data.py --model=1 --sim_type=3 --alpha=.10 --nsim=200 --d=3 --p=3 --q=3 --n=500
nohup python model_generate_data.py --model=1 --sim_type=3 --alpha=.15 --nsim=200 --d=3 --p=3 --q=3 --n=500
nohup python model_generate_data.py --model=1 --sim_type=3 --alpha=.20 --nsim=200 --d=3 --p=3 --q=3 --n=500

# # use Rscript to execute the CLZ test and the KCI test
nohup Rscript model_CLZ.R --model=1 --sim_type=3 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s3_a00.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=3 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s3_a05.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=3 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s3_a10.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=3 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s3_a15.txt &
nohup Rscript model_CLZ.R --model=1 --sim_type=3 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --n_cpu=40 --bandwidth=.2 &> model1_CLZ_s3_a20.txt &

# get result for CLZ and KCI
Rscript model_CLZ_result.R --model=1 --sim_type=3 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.00
Rscript model_CLZ_result.R --model=1 --sim_type=3 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.05
Rscript model_CLZ_result.R --model=1 --sim_type=3 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.10
Rscript model_CLZ_result.R --model=1 --sim_type=3 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.15
Rscript model_CLZ_result.R --model=1 --sim_type=3 --n=500 --p=3 --q=3 --d=3 --bandwidth=.2 --alpha=.20

# # python to execute our proposed test, CDC test, FCIT test
nohup python -u model.py --model=1 --sim_type=3 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 &> model1_s3_a00.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 &> model1_s3_a05.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 &> model1_s3_a10.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 &> model1_s3_a15.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 &> model1_s3_a20.txt &

python model_result.py --model=1 --sim_type=3 --alphas="0.0,0.05,0.1,0.15,0.2" --n=500 --p=3 --q=3 --d=3 --hidden_num=64 

# # conda activate py37
# # python code to execute GCIT
nohup python -u model_GCIT.py --model=1 --sim_type=3 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=50 --cpu=000-040 --nsim=200 --test_prop=.02 &> model1_s3_GCIT_a00.txt &
nohup python -u model_GCIT.py --model=1 --sim_type=3 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=50 --cpu=040-080 --nsim=200 --test_prop=.02 &> model1_s3_GCIT_a05.txt &
nohup python -u model_GCIT.py --model=1 --sim_type=3 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=50 --cpu=080-120 --nsim=200 --test_prop=.02 &> model1_s3_GCIT_a10.txt &
nohup python -u model_GCIT.py --model=1 --sim_type=3 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=50 --cpu=120-160 --nsim=200 --test_prop=.02 &> model1_s3_GCIT_a15.txt &
nohup python -u model_GCIT.py --model=1 --sim_type=3 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=50 --cpu=160-200 --nsim=200 --test_prop=.02 &> model1_s3_GCIT_a20.txt &

python model_result.py --model=1 --sim_type=3 --alphas="0.0,0.05,0.1,0.15,0.2" --n=500 --p=3 --q=3 --d=3 --hidden_num=64 --GCIT=True --GCIT_test_prop=.02

# Generate plots
python model_plot.py --model=1 --sim_type=3 --n=500 --alphas="0.0,0.05,0.1,0.15,0.2" --legend=True