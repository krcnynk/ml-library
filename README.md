# ml-library
This is a small CUDA based ML library
## How to run
Please install Docker and use the supplied Dockerfile to create a container, your current dir must be the project\
* docker build -t my_app . \
* docker run --gpus all -it --rm --shm-size=10.12gb --name my-container -v ${PWD}:/workspace my_app /bin/bash \
To compile in debug mode:\
* cd ./build \
* run docker \
* cmake .. \
* You can run gtest by ./tests/my_cuda_tests

