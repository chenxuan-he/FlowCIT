# conda activate py37
# the conda environment py37 is used to run DGCIT, see its GitHub repo for detailed requirement
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
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> sim_n1000_s1_0.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=5 --nsim=100 &> sim_n1000_s1_2.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=5 --nsim=100 &> sim_n1000_s1_1.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> sim_n1000_s1_-1.txt &
nohup python -u sim.py --sim_type=1 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n1000_s1_-2.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 &> sim_n1000_s2_0.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.05 --par_task=5 --nsim=100 &> sim_n1000_s2_-1.txt &
nohup python -u sim.py --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=-.1 --par_task=5 --nsim=100 &> sim_n1000_s2_-2.txt &
nohup python -u sim.py --p=3 --q=3 --d=3 --n=1000 --alpha=.1 --par_task=100 --nsim=100 &> sim_n1000_s2_2.txt &
nohup python -u sim.py --p=3 --q=3 --d=3 --n=1000 --alpha=.05 --par_task=100 --nsim=100 &> sim_n1000_s2_1.txt &
# try sim 2 H_0 under different hidden_num, similar results
nohup python -u sim.py --gpu=0 --cpu=0-100 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> sim_n1000_s2_0_1.txt &
nohup python -u sim.py --gpu=1 --cpu=100-200 --sim_type=2 --p=3 --q=3 --d=3 --n=1000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=128 &> sim_n1000_s2_0_2.txt &


# seris of (d_x, d_y, d_z, n)=(3, 3, 100, 1000) on server 19
nohup python -u sim.py --gpu=2 --cpu=0-20 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=0 --par_task=5 --nsim=100 &> sim_n1000_s1_d100_0.txt &
nohup python -u sim.py --gpu=3 --cpu=40-60 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=.2 --par_task=5 --nsim=100 &> sim_n1000_s1_d100_2.txt &

# running on 111
nohup python -u sim.py --gpu=5 --cpu=0-50 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=128 &> sim_n1000_s1_d100_0.txt &
nohup python -u sim.py --gpu=6 --cpu=50-100 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=.2 --par_task=5 --nsim=100 --hidden_num=128 &> sim_n1000_s1_d100_2.txt &
nohup python -u sim.py --gpu=7 --cpu=100-150 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=.4 --par_task=5 --nsim=100 --hidden_num=128 &> sim_n1000_s1_d100_4.txt &
nohup python -u sim.py --gpu=0 --cpu=150-200 --sim_type=1 --p=3 --q=3 --d=100 --n=2000 --alpha=0 --par_task=5 --nsim=100 --hidden_num=32 &> sim_n1000_s1_d100_0_32.txt &
