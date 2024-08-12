# Use the official NVIDIA CUDA base image with Ubuntu
# docker pull nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and update the package list
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libcupti-dev \
    libgtest-dev\
    && rm -rf /var/lib/apt/lists/*
# Install CUDA Toolkit and libraries (including cuBLAS, cuFFT, etc.)
RUN apt-get update && apt-get install -y \
    # cuda-toolkit-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN (optional, if you need deep learning libraries)
# Ensure you download the correct version of cuDNN from NVIDIA's website and add it manually
# COPY cudnn-linux-x86_64-*-cuda12.2-*.tgz /tmp/
# RUN tar -xzvf /tmp/cudnn-linux-x86_64-*-cuda12.2-*.tgz -C /usr/local && \
#     rm /tmp/cudnn-linux-x86_64-*-cuda12.2-*.tgz

# Set up environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ls /usr/local/cuda/include/cublas_v2.h

# Verify installation
# RUN nvcc --version && \
#     nvidia-smi

# Set the working directory
VOLUME /workspace
WORKDIR /workspace

# Copy your application code to the container
# COPY . /workspace

# Set default command to bash
# CMD ["/bin/bash"]

# docker build -t my_app .
# docker run --gpus all -it --rm --shm-size=10.12gb --name my-container -v ${PWD}:/workspace my_app /bin/bash
# docker run -it --rm --name my-container -v ${PWD}:/workspace my_app bash
