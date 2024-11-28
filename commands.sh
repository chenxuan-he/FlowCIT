#!/bin/bash
nohup python3 flow_test_gpu.py --p=1 --q=1 --d=1 
nohup python3 flow_test_gpu.py --p=5 --q=5 --d=1 
nohup python3 flow_test_gpu.py --p=10 --q=10 --d=3 
nohup python3 flow_test_gpu.py --p=100 --q=100 --d=20 --n=5000 