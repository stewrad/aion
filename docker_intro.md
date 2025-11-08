# Docker Basic Commands: 
## 1. Run a container: 
```zsh
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  sionna_gpu_env
```
- `docker run` starts a new container
- `-it` is for interactive terminal
- `--rm` auto-removes container when it stops
- `--gpus all` exposes all GPUs (req NVIDIA toolkit)
- `-v $(pwd):/workspace` mounts your current dir into /workspace
- `p 8888:8888` maps Jupyter or web ports
- `sionna_gpu_env` is name of the image

Run in the background:
```zsh
docker run -d --name sionna_container sionna_gpu_env
```
Connect to it later: 
```zsh
docker exec -it sionna_container bash
```
Stop it when done:
```zsh
docker stop sionna_container
```

## 2. List Containers

Running containers:
```zsh
docker ps
```
All containers (including stopped):
```zsh
docker ps -a
```

## 3. Stop & Remove containers 

Stop one container:
```zsh
docker stop <container_id_or_name>
```
Remove one container:
```zsh
docker rm <container_id_or_name>
```
Remove all stopped containers:
```zsh
docker container prune
```

## 4. Manage Images

- List images:
```zsh
docker images
```
- Remove one image:
```zsh
docker rmi <image_id_or_name>
```
- Remove unused images:
```zsh
docker image prune
```
Remove *everything* unused (containers, images, volumnes, networks):
```zsh
docker system prune -a
```
- **this is nuclear, it will wipe all unreferences images and stopped containers**

## 5. Persistent Storage

If you want data to persist beyond ocntainer deletion, **mount volumes**:
```zsh
docker run -it -v ~/projects/sionna:/workspace sionna_gpu_env
```
- now anything in `/workspace` is actually stored in `~/projects/sionna`

## 6. Inspect and Logs: 

View container logs:
```zsh
docker logs <container_id_or_name>
```
Inspect full and container metadata:
```zsh
docker inspect <container_id_or_name>
```

## 7. Quick Cleanup Cheat Sheet

One-liner to cleanup everything safely:
```zsh
docker stop $(docker ps -aq) 2>/dev/null
docker system prune -a -f --volumes
```

## 8. Rebuild Image after editing Dockerfile

If you modify `Dockerfile`, rebuild:
```zsh
docker build -t sionna_gpu_env .

# then 
docker run -it --gpus all --rm sionna_gpu_env
```

# Docker setup info

## 1. Create a project directory
```zsh
mkdir docker_dir
cd docker_dir
```

## 2. Create a dockerfile

Minimal but robust GPU-compatible build (dockerfile):
```dockerfile
# Use a base image with Ubuntu + CUDA + cuDNN + Python 3.12
# (Assuming NVIDIA maintains such an image — adjust tag as needed)
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3   
# as placeholder

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
        git \
        vim \
        python3.12 python3.12-dev python3.12-venv \
        && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3 -m ensurepip \
    && pip3 install --upgrade pip setuptools wheel

# Install TensorFlow　(choose a version compatible with Sionna)
# According to Sionna docs: TF 2.14-2.19 recommended. We’ll pick 2.15.0 as an example.
RUN pip install tensorflow==2.15.0

# Install Sionna (latest) via pip
RUN pip install sionna

# Optionally install JupyterLab and other useful tools
RUN pip install jupyterlab numpy matplotlib scipy

# Expose port for Jupyter (if you want to run Jupyter inside)
EXPOSE 8888

# Default command
CMD ["bash"]
```

## 3. Build the image 
```zsh
docker build -t sionna_env . 
```

This downloads the NVIDIA TensorFlow base (~3GB), then adds Sionna and Tools 

## 4. Run the container

If you have an NVIDIA GPU and the `nvidia-container-toolkit` installed: 
```zsh
docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/workspace sionna_env
```

If no GPU: 
```zsh
docker run -it --rm -p 8888:8888 -v $(pwd):/workspace sionna_env
```

# Alternative - Optimize Docker setup for Manjaro with GPU 
## 1. Install NVIDIA Container Toolkit on Manjaro
```zsh
# Ensure you have Docker installed
sudo pacman -S docker

# Enable and start Docker
sudo systemctl enable --now docker

# Add your user to the docker group (then log out/in)
sudo usermod -aG docker $USER
```

Now install NVIDIA toolkit:
```zsh
# Install the nvidia-container-toolkit (from the Arch community repo)
sudo pacman -S nvidia-container-toolkit
```

Then configure Docker to use it: 
```zsh
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify its working:
```zsh
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

## 2. Create project and Dockerfile
```zsh
mkdir ~/sionna_docker && cd ~/sionna_docker
```

Dockerfile
```dockerfile
# =====================================================
# GPU-Optimized Sionna Environment for Manjaro + RTX 5080
# =====================================================

# Use NVIDIA TensorFlow container (includes CUDA/cuDNN + Python 3.10)
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

# Set working directory
WORKDIR /workspace

# Upgrade pip and install compatible versions
RUN pip install --upgrade pip setuptools wheel && \
    pip install sionna==0.15.0 tensorflow==2.15.0 && \
    pip install jupyterlab numpy matplotlib scipy && \
    pip cache purge

# Optional: install developer tools
RUN apt-get update && apt-get install -y vim git && apt-get clean

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["bash"]
```

## 3. Build the image
```zsh
docker build -t sionna_gpu_env .
```

## 4. Run the container with GPU access
```zsh
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  sionna_gpu_env
```

## 5. Test Sionna and GPU
```zsh
python

import tensorflow as tf
import sionna

print("TensorFlow version:", tf.__version__)
print("Sionna version:", sionna.__version__)
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))
```


# Docker WITH GNU Radio:

```zsh
xhost +local:docker  # allow docker to use your X server
docker run -it --rm \
    --gpus all \
    --network host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    acm_sim_env-gr 
```