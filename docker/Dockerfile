# Docker config file for building image for Group-Diffusion
# This config file is based on that provided in (https://github.com/openai/consistency_models)
#
# Date  : 2024-04-04
#
# Note, setup.py is not longer supported by setuptools > v58.2?

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libopenmpi-dev git wget \
    python3 python3-dev python3-pip python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc\
    && echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc

RUN pip3 install --no-cache-dir setuptools==58.2.0 wheel packaging mpi4py 
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 
RUN pip3 install --no-cache-dir flash-attn==0.2.8
RUN pip3 install blobfile scipy piq tqdm numpy pillow clean-fid==0.1.35

# Install project 
WORKDIR /home/
RUN git clone https://github.com/SpencerSzabados/Group-Diffusion-Bridge.git
WORKDIR /home/Group-Diffusion-Bridge

RUN ln -s /usr/bin/python3 /usr/bin/python
