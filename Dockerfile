FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# upgrade existing stuff
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# install packages for development
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ssh \
    software-properties-common build-essential \
    gcc g++ gdb clang cmake rsync tar git \
    python3-dev python3-pip apt-utils \
    sudo vim zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev \
    wget apt-utils python-is-python3 zip unzip \
    && apt-get clean

# packages needed when container wants to use host S server
RUN apt-get install -y xdg-user-dirs xdg-utils

# setup pytorch
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip --directory-prefix /opt/ && \
    sudo unzip /opt/libtorch-cxx11*  -d /opt/

# create non-root user rr
RUN adduser --disabled-password --gecos '' rr
RUN adduser rr sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER rr
WORKDIR /home/rr
RUN echo "export USER=rr" >> ~/.bashrc
ENTRYPOINT /bin/bash

# upgrade python pip
RUN pip install --upgrade pip

# install required python packages
COPY requirements.txt /tmp/requirements.txt
WORKDIR /tmp/
RUN pip install -r requirements.txt

# set final workdir
WORKDIR /home/rr