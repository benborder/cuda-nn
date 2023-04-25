# CUDA Neural Network

This is a simple neural network built using CUDA for matrix operations.

## Features

- [x] Fully connected (MLP) network
- [ ] Convolutional network
- [ ] Optimisers
	- [x] Stochastic Gradient Descent (SGD)
	- [x] Gradient normalised clipping
	- [ ] SGD with momentum
	- [ ] Adam
- [x] 2D matrix operations
- [ ] 3D matrix operations (to include batch)

## Dependencies

The library has been designed to have as few dependencies as possible with the only dependencies for the core library:

- [cuda](https://developer.nvidia.com/cuda-toolkit) (build and runtime)
- [spdlog](https://github.com/gabime/spdlog) (build)
- [MNIST](http://yann.lecun.com/exdb/mnist) (runtime)
- Compiler with C++20 support
- CMake 3.22 or newer

MNIST and spdlog are automatically fetched via CMake.

## Building

Build via CMake

```bash
cmake --preset release
cmake --build --preset release --target cuda-nn --parallel 8
```
