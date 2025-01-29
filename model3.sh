# # conda activate py37
# # the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# # model 2: (d_x, d_y, d_z, n)=(50, 50, 100, 1000), linear sim_type=1
nohup python -u model.py --gpu=7 --cpu=0-220 --model=3 --sim_type=1 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model4_s1_d3_0.txt &
nohup python -u model.py --gpu=6 --cpu=0-220 --model=3 --sim_type=1 --d=50 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> model4_s1_d50_0.txt &

nohup python -u model.py --gpu=5 --cpu=0-220 --model=3 --sim_type=1 --d=3 --n=1000 --alpha=0.5 --par_task=5 --nsim=100 --hidden_num=32 &> model4_s1_d3_5.txt &
nohup python -u model.py --gpu=4 --cpu=0-220 --model=3 --sim_type=1 --d=50 --n=1000 --alpha=0.5 --par_task=5 --nsim=100 --hidden_num=32 &> model4_s1_d50_5.txt &

