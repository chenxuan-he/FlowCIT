# conda activate py37
# the conda environment py37 is used to run DGCIT, see its GitHub repo https://github.com/tianlinxu312/dgcit for detailed requirements.
# model 1: (d_x, d_y, d_z, n)=(3, 3, 3, 500)
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=0 --par_task=5 --nsim=100 &> model1_n500_s1_0.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> model1_n500_s1_2.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.1 --par_task=5 --nsim=100 &> model1_n500_s1_1.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=-.1 --par_task=5 --nsim=100 &> model1_n500_s1_-1.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=-.2 --par_task=5 --nsim=100 &> model1_n500_s1_-2.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=0 --par_task=5 --nsim=100 &> model1_n500_s2_0.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> model1_n500_s2_2.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.1 --par_task=5 --nsim=100 &> model1_n500_s2_1.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=-.1 --par_task=5 --nsim=100 &> model1_n500_s2_-1.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=-.2 --par_task=5 --nsim=100 &> model1_n500_s2_-2.txt &

# model 1: (d_x, d_y, d_z, n)=(3, 3, 3, 1000)
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> model1_n1000_s1_0.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=5 --nsim=100 &> model1_n1000_s1_2.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=5 --nsim=100 &> model1_n1000_s1_1.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> model1_n1000_s1_-1.txt &
nohup python -u model.py --model=1 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> model1_n1000_s1_-2.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> model1_n1000_s2_0.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> model1_n1000_s2_-1.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> model1_n1000_s2_-2.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=100 --nsim=100 &> model1_n1000_s2_2.txt &
nohup python -u model.py --model=1 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=100 --nsim=100 &> model1_n1000_s2_1.txt &
