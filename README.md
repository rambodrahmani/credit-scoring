# Machine Learning for credit scoring
Machine Learning and Deep Learning models for credit default prediction.

## Setup
The provided Docker file can be used to setup a container with everything that is needed to reproduce the presented experiments:
```bash
docker build -t credit-scoring:latest -f Dockerfile . 
```

Make sure to update the `-v /home/rr/DevOps/:/home/rr/DevOps` parameter and run the container for the first time using:
```bash
docker run --gpus 'all,"capabilities=graphics,utility,display,video,compute"' --net host --privileged --name credit-scoring -itu rr -e NVIDIA_VISIBLE_DEVICES=all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /home/rr/DevOps:/home/rr/DevOps credit-scoring /bin/bash 
```

If you get errors like `Error: cannot open display`, try fixing it by running
```bash
xhost local:root 
```

In order to run the Jupyter notebooks, `src` must be in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/rr/DevOps/credit-scoring"
```

Create models directory:
```bash
mkdir models
```

Download datasets:
```bash

```

Start the Jupyter Lab:
```bash
cd /home/rr/DevOps/credit-scoring
~/.local/bin/jupyter lab --no-browser --ip "*"
```