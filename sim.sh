# conda activate py37
# series of (d_x, d_y, d_z, n)=(3, 3, 3, 500)
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=0 --par_task=5 --nsim=100 &> sim_n500_s1_0.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> sim_n500_s1_2.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=.1 --par_task=5 --nsim=100 &> sim_n500_s1_1.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n500_s1_-1.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=500 --alpha=-.2 --par_task=5 --nsim=100 &> sim_n500_s1_-2.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=0 --par_task=5 --nsim=100 &> sim_n500_s2_0.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.2 --par_task=5 --nsim=100 &> sim_n500_s2_2.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=.1 --par_task=5 --nsim=100 &> sim_n500_s2_1.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n500_s2_-1.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=500 --alpha=-.2 --par_task=5 --nsim=100 &> sim_n500_s2_-2.txt &

# series of (d_x, d_y, d_z, n)=(3, 3, 3, 1000)
nohup python -u sim.py --gpu=3 --cpu=80-120 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> sim_n1000_s1_0.txt &
nohup python -u sim.py --gpu=5 --cpu=120-160 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=5 --nsim=100 &> sim_n1000_s1_2.txt &
nohup python -u sim.py --gpu=6 --cpu=160-192 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=5 --nsim=100 &> sim_n1000_s1_1.txt &
nohup python -u sim.py --gpu=7 --cpu=192-224 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> sim_n1000_s1_-1.txt &
nohup python -u sim.py --gpu=0 --cpu=0-40 --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n1000_s1_-2.txt &
nohup python -u sim.py --gpu=1 --cpu=40-80 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> sim_n1000_s2_0.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=5 --nsim=100 &> sim_n1000_s2_2.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=5 --nsim=100 &> sim_n1000_s2_1.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> sim_n1000_s2_-1.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n1000_s2_-2.txt &


