nohup python -u model.py --model=2 --sim_type=1 --alpha=.0 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=000-020 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model2_s1_a00.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.2 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=2 --cpu=020-040 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model2_s1_a20.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.4 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=040-060 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model2_s1_a40.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.6 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=060-080 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model2_s1_a60.txt &
nohup python -u model.py --model=2 --sim_type=1 --alpha=.8 --n=1000 --p=50 --q=50 --d=100 --par_task=5 --gpu=3 --cpu=080-100 --nsim=200 --CCIT=0 --CDC=0 --FCIT=0 --FlowCIT=1 --FlowCIT_method="IPC" --FlowCIT_permutation=0 &> model2_s1_a80.txt &