nohup python -u model.py --model=1 --sim_type=3 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 --hidden_num=16 --CDC=0 --FCIT=0 &> model1_s3_a00.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 --hidden_num=16 --CDC=0 --FCIT=0 &> model1_s3_a05.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 --hidden_num=16 --CDC=0 --FCIT=0 &> model1_s3_a10.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=140-180 --nsim=200 --hidden_num=16 --CDC=0 --FCIT=0 &> model1_s3_a15.txt &
nohup python -u model.py --model=1 --sim_type=3 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=5 --cpu=180-220 --nsim=200 --hidden_num=16 --CDC=0 --FCIT=0 &> model1_s3_a20.txt 

nohup python -u model.py --model=1 --sim_type=4 --alpha=.00 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=1 --cpu=000-040 --nsim=200 --hidden_num=64 &> model1_s4_a00.txt &
nohup python -u model.py --model=1 --sim_type=4 --alpha=.05 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=2 --cpu=040-080 --nsim=200 --hidden_num=64 &> model1_s4_a05.txt &
nohup python -u model.py --model=1 --sim_type=4 --alpha=.10 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=3 --cpu=080-120 --nsim=200 --hidden_num=64 &> model1_s4_a10.txt &
nohup python -u model.py --model=1 --sim_type=4 --alpha=.15 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=4 --cpu=120-160 --nsim=200 --hidden_num=64 &> model1_s4_a15.txt &
nohup python -u model.py --model=1 --sim_type=4 --alpha=.20 --n=500 --p=3 --q=3 --d=3 --par_task=5 --gpu=5 --cpu=160-200 --nsim=200 --hidden_num=64 &> model1_s4_a20.txt &
