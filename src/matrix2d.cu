#include "helpers.h"
#include "matrix2d.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace cg = cooperative_groups;

namespace
{
constexpr int ktile_dim = 16;
} // namespace

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

__global__ void cuda_add(float* mat1, float* mat2, float* mat3, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat3[x] = mat1[x] + mat2[x];
}

__global__ void cuda_add_self(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat1[x] += mat2[x];
}

__global__ void cuda_sub(float* mat1, float* mat2, float* mat3, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat3[x] = mat1[x] - mat2[x];
}

__global__ void cuda_sub_self(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat1[x] -= mat2[x];
}

__global__ void cuda_mul_cwise(float* mat1, float* mat2, float* mat3, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat3[x] = mat1[x] * mat2[x];
}

__global__ void cuda_mul_cwise_self(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat1[x] *= mat2[x];
}

__global__ void cuda_mat_mul(float* mat1, float* mat2, float* mat3, int nx, int ny, int m)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	const int my = m * y;
	float dot_prod = 0.0F;
	for (int i = 0; i < m; ++i) { dot_prod += mat1[my + i] * mat2[nx * i + x]; }

	mat3[nx * y + x] = dot_prod;
}

__global__ void cuda_div_cwise(float* mat1, float* mat2, float* mat3, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat3[x] = mat1[x] / mat2[x];
}

__global__ void cuda_div_cwise_self(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat1[x] /= mat2[x];
}

__global__ void cuda_max_cwise(float* mat1, float* mat2, float* mat3, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat3[x] = mat1[x] > mat2[x] ? mat1[x] : mat2[x];
}

__global__ void cuda_add_scalar(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = mat1[x] + scalar;
}

__global__ void cuda_add_scalar_self(float* mat, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat[x] += scalar;
}

__global__ void cuda_sub_scalar(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = mat1[x] - scalar;
}

__global__ void cuda_sub_scalar_self(float* mat, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat[x] -= scalar;
}

__global__ void cuda_mul_scalar(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = mat1[x] * scalar;
}

__global__ void cuda_mul_scalar_self(float* mat, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat[x] *= scalar;
}

__global__ void cuda_invert_scalar(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = scalar / mat1[x];
}

__global__ void cuda_scalar_sub(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = scalar - mat1[x];
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

__global__ void cuda_norm_len(float* mat, float* sum, int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	float* s_data = SharedMemory<float>();

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	float tsum = (i < n) ? mat[i] * mat[i] : 0.0F;
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

__global__ void cuda_isnan(float* mat, bool* is_nan, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	// We dont care about race conditions as any write is going to be true
	if (isnan(mat[x])) { *is_nan = true; }
}

__global__ void cuda_isinf(float* mat, bool* is_inf, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	// We dont care about race conditions as any write is going to be true
	if (isinf(mat[x])) { *is_inf = true; }
}

__global__ void cuda_fill(float* mat, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat[x] = scalar;
}

__global__ void cuda_eye(float* mat1, int nx, int ny)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	mat1[nx * y + x] = x == y ? 1.0F : 0.0F;
}

__global__ void cuda_exp(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = std::exp(mat1[x]);
}

__global__ void cuda_neg(float* mat1, float* mat2, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = -mat1[x];
}

__global__ void cuda_cwise_max(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	if (scalar > mat1[x]) { mat2[x] = scalar; }
}

__global__ void cuda_cwise_gt(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = mat1[x] > scalar ? 1.0F : 0.0F;
}

__global__ void cuda_cwise_lt(float* mat1, float* mat2, float scalar, int n)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= n) { return; }
	mat2[x] = mat1[x] < scalar ? 1.0F : 0.0F;
}

__global__ void cuda_transpose(float* mat_in, float* mat_out, int nx, int ny)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= nx) || (y >= ny)) { return; }
	mat_out[x * ny + y] = mat_in[y * nx + x];
}

template <typename T = float>
inline std::shared_ptr<T> make_shared_cuda(size_t num_elements)
{
	float* device_ptr = nullptr;
	CHECK_CUDA_ERROR(cudaMalloc(&device_ptr, num_elements * sizeof(T)));
	return std::shared_ptr<T>(device_ptr, [](T* dev_ptr) { CHECK_CUDA_ERROR(cudaFree(dev_ptr)); });
}

Matrix2d::Matrix2d(Size size) : size_(size), num_elements_(size_.x * size_.y)
{
	assert(size_.x > 0);
	assert(size_.y > 0);
	set_block_thread_size();
	d_data_ = make_shared_cuda(num_elements_);
}

Matrix2d::Matrix2d(const std::vector<float>& data, Size size) : size_(size)
{
	assert(data.size() > 0);
	if (size.y < 0)
	{
		size.x = 1;
		size.y = static_cast<int>(data.size());
	}
	else
	{
		assert(size.x > 0);
		assert(size.y > 0);
		assert(data.size() == static_cast<size_t>(size.x * size.y));
	}

	set_block_thread_size();

	d_data_ = make_shared_cuda(data.size());
	set(data, size);
}

Matrix2d::Matrix2d(Size size, float scalar) : size_(size), num_elements_(size_.x * size_.y)
{
	assert(size_.x > 0);
	assert(size_.y > 0);
	set_block_thread_size();

	d_data_ = make_shared_cuda(num_elements_);
	fill(scalar);
}

Matrix2d::Matrix2d(float scalar) : size_({1, 1}), num_elements_(1)
{
	set_block_thread_size();
	d_data_ = make_shared_cuda(num_elements_);
	fill(scalar);
}

Matrix2d::Matrix2d(const Matrix2d& mat)
		: size_(mat.size_)
		, num_elements_(mat.num_elements_)
		, blocks_(mat.blocks_)
		, threads_(mat.threads_)
		, d_data_(mat.d_data_)
{
}

Matrix2d::Matrix2d(Matrix2d&& mat)
		: d_data_(std::move(mat.d_data_))
		, size_(mat.size_)
		, num_elements_(mat.num_elements_)
		, blocks_(mat.blocks_)
		, threads_(mat.threads_)
{
	mat.d_data_ = nullptr;
	mat.size_ = {0, 0};
	mat.num_elements_ = 0;
}

Matrix2d::~Matrix2d()
{
	num_elements_ = 0;
}

Matrix2d Matrix2d::clone() const
{
	Matrix2d mat(size_);
	CHECK_CUDA_ERROR(
		cudaMemcpy(mat.d_data_.get(), d_data_.get(), num_elements_ * sizeof(float), cudaMemcpyDeviceToDevice));
	return mat;
}

void Matrix2d::set_block_thread_size()
{
	threads_ = dim3(ktile_dim, ktile_dim);
	blocks_ = dim3((size_.x + threads_.x - 1) / threads_.x, (size_.y + threads_.y - 1) / threads_.y);
}

Matrix2d& Matrix2d::operator=(const Matrix2d& mat)
{
	size_ = mat.size_;
	num_elements_ = mat.num_elements_;
	threads_ = mat.threads_;
	blocks_ = mat.blocks_;
	d_data_ = mat.d_data_;
	return *this;
}

Matrix2d& Matrix2d::operator=(Matrix2d&& mat)
{
	size_ = mat.size_;
	num_elements_ = mat.num_elements_;
	blocks_ = mat.blocks_;
	threads_ = mat.threads_;
	d_data_ = std::move(mat.d_data_);
	return *this;
}

Size Matrix2d::size() const
{
	return size_;
}

int Matrix2d::num_elements() const
{
	return num_elements_;
}

void Matrix2d::check_bounds_match(const Matrix2d& mat) const
{
	if (mat.size_.x != size_.x)
	{
		std::cerr << "x dims do not match. Expected " << size_.x << " but got " << mat.size_.x << std::endl;
		throw std::runtime_error("Dim mismatch");
	}
	if (mat.size_.y != size_.y)
	{
		// y dims to not match
		std::cerr << "y dims do not match. Expected " << size_.y << " but got " << mat.size_.y << std::endl;
		throw std::runtime_error("Dim mismatch");
	}
}

Matrix2d Matrix2d::add(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_add, d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::sub(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_sub, d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::mul(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_mul_cwise, d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::mmul(const Matrix2d& mat) const
{
	if (size_.x != mat.size_.y)
	{
		std::cerr << "x dim of mat1 does not match y dim of mat2. Expected " << size_.x << " but got " << mat.size_.y
							<< std::endl;
		throw std::runtime_error("Dim mismatch");
	}

	Matrix2d mat_result({size_.y, mat.size_.x});

	cuda_mat_mul<<<mat_result.blocks_, mat_result.threads_>>>(
		d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), mat.size_.x, size_.y, size_.x);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());

	return mat_result;
}

Matrix2d Matrix2d::div(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_div_cwise, d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::max(const Matrix2d& mat) const
{
	check_bounds_match(mat);

	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_max_cwise, d_data_.get(), mat.d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::add(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_add_scalar, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::sub(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_sub_scalar, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::mul(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_mul_scalar, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::div(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_mul_scalar, d_data_.get(), mat_result.d_data_.get(), 1.0F / scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::max(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_cwise_max, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::gt(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_cwise_gt, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::lt(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_cwise_lt, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::inv(const float scalar) const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_invert_scalar, d_data_.get(), mat_result.d_data_.get(), scalar, num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::operator+(const Matrix2d& mat) const
{
	return add(mat);
}

Matrix2d& Matrix2d::operator+=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL_1D(cuda_add_self, d_data_.get(), mat.d_data_.get(), num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator-(const Matrix2d& mat) const
{
	return sub(mat);
}

Matrix2d Matrix2d::operator-() const
{
	return neg();
}

Matrix2d& Matrix2d::operator-=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL_1D(cuda_sub_self, d_data_.get(), mat.d_data_.get(), num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator*(const Matrix2d& mat) const
{
	return mmul(mat);
}

Matrix2d& Matrix2d::operator*=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL_1D(cuda_mul_cwise_self, d_data_.get(), mat.d_data_.get(), num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator/(const Matrix2d& mat) const
{
	return div(mat);
}

Matrix2d& Matrix2d::operator/=(const Matrix2d& mat)
{
	check_bounds_match(mat);

	KERNEL_CALL_1D(cuda_div_cwise_self, d_data_.get(), mat.d_data_.get(), num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator+(const float scalar) const
{
	return add(scalar);
}

Matrix2d& Matrix2d::operator+=(const float scalar)
{
	KERNEL_CALL_1D(cuda_add_scalar_self, d_data_.get(), scalar, num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator-(const float scalar) const
{
	return sub(scalar);
}

Matrix2d& Matrix2d::operator-=(const float scalar)
{
	KERNEL_CALL_1D(cuda_sub_scalar_self, d_data_.get(), scalar, num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator*(const float scalar) const
{
	return mul(scalar);
}

Matrix2d& Matrix2d::operator*=(const float scalar)
{
	KERNEL_CALL_1D(cuda_mul_scalar_self, d_data_.get(), scalar, num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator/(const float scalar) const
{
	return div(scalar);
}

Matrix2d& Matrix2d::operator/=(const float scalar)
{
	KERNEL_CALL_1D(cuda_mul_scalar_self, d_data_.get(), 1.0F / scalar, num_elements_);

	return *this;
}

Matrix2d Matrix2d::operator>(const float scalar) const
{
	return gt(scalar);
}

Matrix2d Matrix2d::operator<(const float scalar) const
{
	return lt(scalar);
}

Matrix2d operator-(float scalar, const Matrix2d& mat)
{
	Matrix2d mat_result(mat.size_);

	const dim3 dim_block(mat.threads_.x * mat.threads_.y * mat.threads_.z, 1, 1);
	const dim3 dim_grid(mat.blocks_.x * mat.blocks_.y * mat.blocks_.z, 1, 1);
	cuda_scalar_sub<<<dim_block, dim_grid>>>(mat.d_data_.get(), mat_result.d_data_.get(), scalar, mat.num_elements_);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());

	return mat_result;
	return mat.neg() + scalar;
}

float Matrix2d::sum() const
{
	float* d_sum;
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_sum, sizeof(float)));
	const dim3 dim_block(threads_.x * threads_.y * threads_.z, 1, 1);
	const dim3 dim_grid(blocks_.x * blocks_.y * blocks_.z, 1, 1);
	const int smem_size = (dim_block.x <= 32) ? 2 * dim_block.x * sizeof(float) : dim_block.x * sizeof(float);
	cuda_sum<<<dim_grid, dim_block, smem_size>>>(d_data_.get(), d_sum, num_elements_);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
	float sum = *d_sum;
	CHECK_CUDA_ERROR(cudaFree(d_sum));
	return sum;
}

float Matrix2d::mag() const
{
	float* d_norm_len;
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_norm_len, sizeof(float)));
	const dim3 dim_block(threads_.x * threads_.y * threads_.z, 1, 1);
	const dim3 dim_grid(blocks_.x * blocks_.y * blocks_.z, 1, 1);
	const int smem_size = (dim_block.x <= 32) ? 2 * dim_block.x * sizeof(float) : dim_block.x * sizeof(float);
	cuda_sum<<<dim_grid, dim_block, smem_size>>>(d_data_.get(), d_norm_len, num_elements_);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
	float mag = *d_norm_len;
	CHECK_CUDA_ERROR(cudaFree(d_norm_len));
	if (std::isnan(mag)) { throw std::runtime_error("NAN"); }
	return std::sqrt(mag);
}

Matrix2d Matrix2d::norm() const
{
	return mul(1.0F / mag());
}

Matrix2d Matrix2d::transpose() const
{
	Matrix2d mat_result({size_.x, size_.y});

	KERNEL_CALL_2D(cuda_transpose, d_data_.get(), mat_result.d_data_.get(), size_.x, size_.y);

	return mat_result;
}

Matrix2d Matrix2d::flatten()
{
	Matrix2d mat(*this);
	mat.size_ = Size{num_elements_, 1};
	return mat;
}

bool Matrix2d::isnan() const
{
	bool* d_is_nan;
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_is_nan, sizeof(bool)));
	*d_is_nan = false;
	cudaDeviceSynchronize();
	KERNEL_CALL_1D(cuda_isnan, d_data_.get(), d_is_nan, num_elements_);
	bool is_nan = *d_is_nan;
	CHECK_CUDA_ERROR(cudaFree(d_is_nan));
	return is_nan;
}

bool Matrix2d::isinf() const
{
	bool* d_is_inf;
	CHECK_CUDA_ERROR(cudaMallocManaged((void**)&d_is_inf, sizeof(bool)));
	*d_is_inf = false;
	cudaDeviceSynchronize();
	KERNEL_CALL_1D(cuda_isinf, d_data_.get(), d_is_inf, num_elements_);
	bool is_inf = *d_is_inf;
	CHECK_CUDA_ERROR(cudaFree(d_is_inf));
	return is_inf;
}

void Matrix2d::fill(float scalar)
{
	KERNEL_CALL_1D(cuda_fill, d_data_.get(), scalar, num_elements_);
}

Matrix2d Matrix2d::exp() const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_exp, d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

Matrix2d Matrix2d::neg() const
{
	Matrix2d mat_result(size_);

	KERNEL_CALL_1D(cuda_neg, d_data_.get(), mat_result.d_data_.get(), num_elements_);

	return mat_result;
}

float Matrix2d::get(int y, int x) const
{
	float element;
	int i = size_.x * y + x;
	CHECK_CUDA_ERROR(cudaMemcpy(&element, &(d_data_.get()[i]), sizeof(float), cudaMemcpyDeviceToHost));
	return element;
}

std::vector<float> Matrix2d::get() const
{
	std::vector<float> data;
	data.resize(num_elements_);
	CHECK_CUDA_ERROR(cudaMemcpy(data.data(), d_data_.get(), data.size() * sizeof(float), cudaMemcpyDeviceToHost));
	return data;
}

void Matrix2d::set(const std::vector<float>& data, Size size)
{
	assert(size.x > 0);
	assert(size.y > 0);
	size_ = size;
	num_elements_ = static_cast<int>(data.size());
	CHECK_CUDA_ERROR(cudaMemcpy(d_data_.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
}

Matrix2d eye(Size size)
{
	Matrix2d mat(size);
	cuda_eye<<<mat.blocks_, mat.threads_>>>(mat.d_data_.get(), mat.size_.x, mat.size_.y);
	cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaGetLastError());
	return mat;
}
