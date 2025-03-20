# # conda activate py37
# # the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# # model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), linear sim_type=1
nohup python -u model.py --model=2 --sim_type=1 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s1_0.txt &
nohup python -u model.py --model=2 --sim_type=1 --p=50 --q=50 --d=100 --n=1000 --alpha=0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s1_4.txt &
nohup python -u model.py --model=2 --sim_type=1 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s1_-4.txt &
nohup python -u model.py --model=2 --sim_type=1 --p=50 --q=50 --d=100 --n=1000 --alpha=0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s1_8.txt &
nohup python -u model.py --model=2 --sim_type=1 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s1_-8.txt &

# model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), sim_type=2
nohup python -u model.py --model=2 --sim_type=2 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s2_0.txt &
nohup python -u model.py --model=2 --sim_type=2 --p=50 --q=50 --d=100 --n=1000 --alpha=0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s2_4.txt & 
nohup python -u model.py --model=2 --sim_type=2 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s2_-4.txt & 
nohup python -u model.py --model=2 --sim_type=2 --p=50 --q=50 --d=100 --n=1000 --alpha=0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s2_8.txt &
nohup python -u model.py --model=2 --sim_type=2 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s2_-8.txt 

# model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), sim_type=3
nohup python -u model.py --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_0.txt &
nohup python -u model.py --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_4.txt & 
nohup python -u model.py --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=-.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_-4.txt & 
nohup python -u model.py --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_-8.txt & 
nohup python -u model.py --model=2 --sim_type=3 --p=50 --q=50 --d=100 --n=1000 --alpha=0.8 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s3_8.txt & 

# model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), sim_type=4
nohup python -u model.py --model=2 --sim_type=4 --p=50 --q=50 --d=100 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s4_0.txt &
nohup python -u model.py --model=2 --sim_type=4 --p=50 --q=50 --d=100 --n=1000 --alpha=0.2 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s4_2.txt & 
nohup python -u model.py --model=2 --sim_type=4 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.2 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s4_-2.txt & 
nohup python -u model.py --model=2 --sim_type=4 --p=50 --q=50 --d=100 --n=1000 --alpha=0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s4_4.txt &
nohup python -u model.py --model=2 --sim_type=4 --p=50 --q=50 --d=100 --n=1000 --alpha=-0.4 --par_task=5 --nsim=100 --hidden_num=32 &> model2_n1000_s4_-4.txt 
