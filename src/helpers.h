#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <sstream>
#include <stdexcept>

template <typename T>
void check_and_throw_error(T result, const char* const func, const char* const file, const int line)
{
	if (result)
	{
		if (cudaError::cudaErrorInsufficientDriver == result)
		{
			throw std::runtime_error("The graphics driver is not compatible with the required CUDA version.");
		}
		else if (cudaError::cudaErrorOperatingSystem == result)
		{
			throw std::runtime_error("An OS call within the CUDA api failed.");
		}
		else if (cudaError::cudaErrorInitializationError == result)
		{
			throw std::runtime_error("CUDA could not be initialized. Potential unmet minimum hardware requirements.");
		}
		else if (cudaError::cudaErrorUnsupportedPtxVersion == result)
		{
			throw std::runtime_error("CUDA error occurred: cudaErrorUnsupportedPtxVersion.");
		}
		else
		{
			std::stringstream stream;
			stream << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \""
						 << func << "\"";
			throw std::runtime_error(stream.str().c_str());
		}
		cudaDeviceReset();
		exit(99);
	}
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define CHECK_CUDA_ERROR(val) check_and_throw_error((val), #val, __FILENAME__, __LINE__)

#define KERNEL_CALL_1_1(func, ...) \
	func<<<1, 1>>>(__VA_ARGS__);     \
	cudaDeviceSynchronize();         \
	CHECK_CUDA_ERROR(cudaGetLastError());
#define KERNEL_CALL_2D(func, ...)           \
	func<<<blocks_, threads_>>>(__VA_ARGS__); \
	cudaDeviceSynchronize();                  \
	CHECK_CUDA_ERROR(cudaGetLastError());

#define KERNEL_CALL_1D(func, ...)                                                                 \
	func<<<blocks_.x * blocks_.y * blocks_.z, threads_.x * threads_.y * threads_.z>>>(__VA_ARGS__); \
	cudaDeviceSynchronize();                                                                        \
	CHECK_CUDA_ERROR(cudaGetLastError());

constexpr int isqrt(int s)
{
	if (s <= 1) { return s; }

	int x0 = s / 2;
	int x1 = (x0 + (s / x0)) / 2;

	while (x1 < x0)
	{
		x0 = x1;
		x1 = (x0 + s / x0) / 2;
	}
	return x0;
}
