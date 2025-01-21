nohup python -u sim.py --gpu=0 --sim_type=0 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim1.txt &
nohup python -u sim.py --gpu=1 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim2.txt &
nohup python -u sim.py --gpu=2 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim3.txt &
