nohup python -u sim.py --gpu=0 --cpu=0-50 --sim_type=0 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim1.txt &
nohup python -u sim.py --gpu=1 --cpu=50-100 --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim2.txt &
nohup python -u sim.py --gpu=2 --cpu=100-150 --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim3.txt &

nohup python -u sim.py --gpu=6 --cpu=170-210 --sim_type=0 --p=3 --q=3 --d=3 --n=500 --alpha=.5 --par_task=5 --nsim=100 &> sim_n500_0.txt &


nohup python -u sim.py --gpu=5 --cpu=90-130 --sim_type=0 --p=3 --q=3 --d=3 --n=1000 --alpha=.2 --par_task=5 --nsim=100 &> sim_n1000_1.txt &
nohup python -u sim.py --gpu=6 --cpu=130-170 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.2 --par_task=5 --nsim=100 &> sim_n1000_2.txt &
nohup python -u sim.py --gpu=7 --cpu=170-210 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=.2 --par_task=5 --nsim=100 &> sim_n1000_3.txt &
