conda activate py37
nohup python sim.py --gpu=0 --sim_type=0 --p=3 --q=3 --d=3 --n=500 --alpha=0 --cores=5 --nsim=100 &> sim.txt &