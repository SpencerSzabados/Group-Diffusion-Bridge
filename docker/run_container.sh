# File contains a shell command to creat a docker container using the provided docker 
# image <named> with some set gpu paramters and mounted folders.
#


######################################################################################
# Command to build docker image from Dockerfile                                      #
######################################################################################
docker build -t group-diffusion-bridge:latest /Group-Diffusion-Bridge/docker/Dockerfile 


######################################################################################
# Command to create container from existing image with gpu access and mounted drives #
######################################################################################
docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/Group-Diffusion-Bridge/:/home/Group-Diffusion-Bridge -v /home/datasets:/home/datasets -v /home/checkpoints:/home/checkpoints group-diffusion-bridge /bin/bash

docker container run --gpus device=all  --shm-size 24GB --restart=always -it -d  -v /home/Group-Diffusion-Bridge/:/home/Group-Diffusion-Bridge -v /home/datasets:/home/datasets -v /home/checkpoints:/home/checkpoints group-diffusion-bridge /bin/bash
