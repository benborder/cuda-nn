#include "helpers.h"
#include "matrix2d.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <cstring>
#include <iostream>

namespace cg = cooperative_groups;

template <class T>
struct SharedMemory
{
	__device__ inline operator T*()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T*() const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

__global__ void cuda_add(float* mat1, float* mat2, float* mat3, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat3[i] = mat1[i] + mat2[i];
}

__global__ void cuda_add_self(float* mat1, float* mat2, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat1[i] += mat2[i];
}

__global__ void cuda_sub(float* mat1, float* mat2, float* mat3, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat3[i] = mat1[i] - mat2[i];
}

__global__ void cuda_sub_self(float* mat1, float* mat2, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat1[i] -= mat2[i];
}

__global__ void cuda_mul(float* mat1, float* mat2, float* mat3, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat3[i] = mat1[i] * mat2[i];
}

__global__ void cuda_mul_self(float* mat1, float* mat2, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat2[i] *= mat1[i];
}

__global__ void cuda_mat_mul(float* mat1, float* mat2, float* mat3, int nx, int ny, int m)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int nxy = nx * y;
	float dot_prod = 0.0F;
	for (int i = 0; i < m; ++i) { dot_prod += mat1[nxy + i] * mat2[nx * i + x]; }

	mat3[nx * y + x] = dot_prod;
}

__global__ void cuda_add_scalar(float* mat1, float* mat2, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat2[i] = mat1[i] + scalar;
}

__global__ void cuda_add_scalar_self(float* mat, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat[i] += scalar;
}

__global__ void cuda_sub_scalar(float* mat1, float* mat2, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat2[i] = mat1[i] - scalar;
}

__global__ void cuda_sub_scalar_self(float* mat, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat[i] -= scalar;
}

__global__ void cuda_mul_scalar(float* mat1, float* mat2, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat2[i] = mat1[i] * scalar;
}

__global__ void cuda_mul_scalar_self(float* mat, float scalar, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int i = nx * y + x;
	mat[i] *= scalar;
}

__global__ void cuda_sum(float* mat, float* sum, int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	float* s_data = SharedMemory<float>();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	float tsum = (i < n) ? mat[i] : 0.0F;
	s_data[tid] = tsum;
	cg::sync(cta);

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) { s_data[tid] = tsum = tsum + s_data[tid + s]; }

		cg::sync(cta);
	}

	if (tid == 0) { atomicAdd(sum, tsum); }
}

Matrix2d::Matrix2d(int y, int x) : max_x_(x), max_y_(y), num_elements_(x * y)
{
	assert(x > 0);
	assert(y > 0);
	blocks_ = dim3(
		std::min<unsigned int>(64, (x + threads_.x - 1) / threads_.x),
		std::min<unsigned int>(64, (y + threads_.y - 1) / threads_.y));

	CHECK_CUDA_ERROR(cudaMalloc(&d_data_, num_elements_ * sizeof(float)));
}

Matrix2d::Matrix2d(const std::vector<float>& data, int y, int x)
{
	assert(data.size() > 0);
	if (y < 0)
	{
		x = 1;
		y = static_cast<int>(data.size());
	}
	else
	{
		assert(x > 0);
		assert(y > 0);
		assert(data.size() == static_cast<size_t>(x * y));
	}

	blocks_ = dim3(
		std::min<unsigned int>(64, (x + threads_.x - 1) / threads_.x),
		std::min<unsigned int>(64, (y + threads_.y - 1) / threads_.y));

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_, data.size() * sizeof(float)));
	set(data, y, x);
}

Matrix2d::Matrix2d(const Matrix2d& mat)
		: max_x_(mat.max_x_)
		, max_y_(mat.max_y_)
		, num_elements_(mat.num_elements_)
		, blocks_(mat.blocks_)
		, threads_(mat.threads_)
{
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data_, num_elements_ * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(d_data_, mat.d_data_, num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice));
}

Matrix2d::Matrix2d(Matrix2d&& mat)
		: d_data_(mat.d_data_)
		, max_x_(mat.max_x_)
		, max_y_(mat.max_y_)
		, num_elements_(mat.num_elements_)
		, blocks_(mat.blocks_)
		, threads_(mat.threads_)
{
	mat.d_data_ = nullptr;
	mat.max_x_ = 0;
	mat.max_y_ = 0;
	mat.num_elements_ = 0;
}

Matrix2d::~Matrix2d()
{
	CHECK_CUDA_ERROR(cudaFree(d_data_));
	num_elements_ = 0;
}

void Matrix2d::check_bounds_match(const Matrix2d& mat) const
{
	if (mat.max_x_ != max_x_)
	{
		std::cerr << "x dims do not match. Expected " << max_x_ << " but got " << mat.max_x_ << std::endl;
		throw std::runtime_error("Dim mismatch");
	}
	if (mat.max_y_ != max_y_)
	{
		// y dims to not match
		std::cerr << "y dims do not match. Expected " << max_y_ << " but got " << mat.max_y_ << std::endl;
		throw std::runtime_error("Dim mismatch");
	}
}

Matrix2d Matrix2d::add(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(max_y_, max_x_);

	KERNEL_CALL(cuda_add, d_data_, mat.d_data_, mat_result.d_data_, max_x_, max_y_);

	return mat_result;
}

Matrix2d Matrix2d::sub(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(max_y_, max_x_);

	KERNEL_CALL(cuda_sub, d_data_, mat.d_data_, mat_result.d_data_, max_x_, max_y_);

	return mat_result;
}

Matrix2d Matrix2d::mul(const Matrix2d& mat) const
{
	if (max_x_ == mat.max_y_)
	{
		Matrix2d mat_result(max_y_, mat.max_x_);

		KERNEL_CALL(cuda_mat_mul, d_data_, mat.d_data_, mat_result.d_data_, mat.max_x_, max_y_, max_x_);

		return mat_result;
	}
	else
	{
		std::cerr << "x dim of mat1 does not match y dim of mat2. Expected " << max_x_ << " but got " << mat.max_y_
							<< std::endl;
		throw std::runtime_error("Dim mismatch");
	}
}

Matrix2d Matrix2d::add(const float scalar) const
{
	Matrix2d mat_result(max_y_, max_x_);

	KERNEL_CALL(cuda_add_scalar, d_data_, mat_result.d_data_, scalar, max_x_, max_y_);

	return mat_result;
}

Matrix2d Matrix2d::sub(const float scalar) const
{
	Matrix2d mat_result(max_y_, max_x_);

	KERNEL_CALL(cuda_sub_scalar, d_data_, mat_result.d_data_, scalar, max_x_, max_y_);

	return mat_result;
}

Matrix2d Matrix2d::mul(const float scalar) const
{
	Matrix2d mat_result(max_y_, max_x_);

	KERNEL_CALL(cuda_mul_scalar, d_data_, mat_result.d_data_, scalar, max_x_, max_y_);

	return mat_result;
}

Matrix2d Matrix2d::operator+(const Matrix2d& mat) const
{
	return add(mat);
}

Matrix2d& Matrix2d::operator+=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL(cuda_add_self, d_data_, mat.d_data_, max_x_, max_y_);

	return *this;
}

Matrix2d Matrix2d::operator-(const Matrix2d& mat) const
{
	return sub(mat);
}

Matrix2d& Matrix2d::operator-=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL(cuda_sub_self, d_data_, mat.d_data_, max_x_, max_y_);

	return *this;
}

Matrix2d Matrix2d::operator*(const Matrix2d& mat) const
{
	return mul(mat);
}

Matrix2d& Matrix2d::operator*=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL(cuda_mul_self, d_data_, mat.d_data_, max_x_, max_y_);

	return *this;
}

Matrix2d Matrix2d::operator+(const float scalar) const
{
	return add(scalar);
}

Matrix2d& Matrix2d::operator+=(const float scalar)
{
	KERNEL_CALL(cuda_add_scalar_self, d_data_, scalar, max_x_, max_y_);

	return *this;
}

Matrix2d Matrix2d::operator-(const float scalar) const
{
	return sub(scalar);
}

Matrix2d& Matrix2d::operator-=(const float scalar)
{
	KERNEL_CALL(cuda_sub_scalar_self, d_data_, scalar, max_x_, max_y_);

	return *this;
}

Matrix2d Matrix2d::operator*(const float scalar) const
{
	return mul(scalar);
}

Matrix2d& Matrix2d::operator*=(const float scalar)
{
	KERNEL_CALL(cuda_mul_scalar_self, d_data_, scalar, max_x_, max_y_);

	return *this;
}

float Matrix2d::sum() const
{
	float* d_sum;
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_sum, sizeof(float)));
	const dim3 dim_block(threads_.x * threads_.y * threads_.z, 1, 1);
	const dim3 dim_grid(blocks_.x * blocks_.y * blocks_.z, 1, 1);
	const int smem_size = (dim_block.x <= 32) ? 2 * dim_block.x * sizeof(float) : dim_block.x * sizeof(float);
	cuda_sum<<<dim_grid, dim_block, smem_size>>>(d_data_, d_sum, max_x_ * max_y_);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
	float sum = *d_sum;
	CHECK_CUDA_ERROR(cudaFree(d_sum));
	return sum;
}

float Matrix2d::get(int y, int x) const
{
	float element;
	int i = max_x_ * y + x;
	CHECK_CUDA_ERROR(cudaMemcpy(&element, &(d_data_[i]), sizeof(float), cudaMemcpyDeviceToHost));
	return element;
}

std::vector<float> Matrix2d::get() const
{
	std::vector<float> data;
	data.resize(num_elements_);
	CHECK_CUDA_ERROR(cudaMemcpy(data.data(), d_data_, data.size() * sizeof(float), cudaMemcpyDeviceToHost));
	return data;
}

void Matrix2d::set(const std::vector<float>& data, int y, int x)
{
	assert(x > 0);
	assert(y > 0);
	max_x_ = x;
	max_y_ = y;
	num_elements_ = static_cast<int>(data.size());
	CHECK_CUDA_ERROR(cudaMemcpy(d_data_, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
}
