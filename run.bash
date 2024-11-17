#!/bin/bash
# Run below step one by one in terminal
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
docker build -t ppo_pytorch .
docker run --net host --gpus all -it -v /home/$(whoami)/Documents/projects/ppo-pytorch:/app ppo_pytorch


