# conda activate py37
nohup python -u sim.py --gpu=6 --cpu=170-210 --sim_type=0 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> sim_n500_0.txt &
nohup python -u sim.py --gpu=0 --cpu=0-40 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> sim_n500_1.txt &
nohup python -u sim.py --gpu=1 --cpu=40-80 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> sim_n500_2.txt &
