# conda activate py37
# the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), sim_type=3
nohup python -u model.py --gpu=3 --cpu=0-220 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_0.txt &
nohup python -u model.py --gpu=4 --cpu=0-220 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0.2 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_2.txt & 



nohup python -u model.py --gpu=3 --cpu=0-50 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_0.txt &
nohup python -u model.py --gpu=4 --cpu=40-90 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0.2 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_2.txt & 
nohup python -u model.py --gpu=5 --cpu=85-135 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.2 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_-2.txt & 
nohup python -u model.py --gpu=6 --cpu=130-180 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_-4.txt & 
nohup python -u model.py --gpu=7 --cpu=170-220 --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_4.txt &
