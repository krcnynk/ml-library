# ml-library
This is a small CUDA based ML library
## How to run
Please install Docker and use the supplied Dockerfile to create a container, your current dir must be the project\
> docker build -t my_app . \
> docker run -it --rm --name my-container -v ${PWD}:/workspace my_app /bin/bash \
> cd ./build \
> run docker \
> cmake .. \
> You can run gtest by ./tests/my_cuda_tests

