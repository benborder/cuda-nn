#pragma once

#include <vector_types.h>

#include <memory>
#include <vector>

struct Size
{
	int y;
	int x;
};

class Matrix2d
{
public:
	explicit Matrix2d(Size size);
	explicit Matrix2d(Size size, float scalar);
	explicit Matrix2d(float scalar);
	explicit Matrix2d(const std::vector<float>& data, Size size);
	Matrix2d(const Matrix2d& mat);
	Matrix2d(Matrix2d&& mat);

	~Matrix2d();

	Matrix2d& operator=(const Matrix2d& mat);
	Matrix2d& operator=(Matrix2d&& mat);

	Matrix2d clone() const;

	Size size() const;
	int num_elements() const;

	Matrix2d add(const Matrix2d& mat) const;
	Matrix2d sub(const Matrix2d& mat) const;
	Matrix2d mul(const Matrix2d& mat) const;
	Matrix2d mmul(const Matrix2d& mat) const;
	Matrix2d div(const Matrix2d& mat) const;
	Matrix2d max(const Matrix2d& mat) const;

	Matrix2d add(const float scalar) const;
	Matrix2d sub(const float scalar) const;
	Matrix2d mul(const float scalar) const;
	Matrix2d div(const float scalar) const;
	Matrix2d max(const float scalar) const;
	Matrix2d gt(const float scalar) const;
	Matrix2d lt(const float scalar) const;
	Matrix2d inv(const float scalar = 1.0F) const;

	Matrix2d& operator+=(const Matrix2d& mat);
	Matrix2d operator+(const Matrix2d& mat) const;

	Matrix2d& operator-=(const Matrix2d& mat);
	Matrix2d operator-(const Matrix2d& mat) const;
	Matrix2d operator-() const;

	Matrix2d& operator*=(const Matrix2d& mat);
	Matrix2d operator*(const Matrix2d& mat) const;

	Matrix2d& operator/=(const Matrix2d& mat);
	Matrix2d operator/(const Matrix2d& mat) const;

	Matrix2d& operator+=(const float scalar);
	Matrix2d operator+(const float scalar) const;

	Matrix2d& operator-=(const float scalar);
	Matrix2d operator-(const float scalar) const;

	Matrix2d& operator*=(const float scalar);
	Matrix2d operator*(const float scalar) const;

	Matrix2d& operator/=(const float scalar);
	Matrix2d operator/(const float scalar) const;

	Matrix2d operator>(const float scalar) const;
	Matrix2d operator<(const float scalar) const;

	friend Matrix2d operator-(float scalar, const Matrix2d& mat);

	float sum() const;
	float mag() const;
	Matrix2d norm() const;
	Matrix2d exp() const;
	Matrix2d neg() const;
	Matrix2d transpose() const;
	Matrix2d flatten();
	bool isnan() const;
	bool isinf() const;
	friend Matrix2d eye(Size size);

	void fill(float scalar);

	float get(int y, int x) const;
	std::vector<float> get() const;

	void set(const std::vector<float>& data, Size size);

private:
	void check_bounds_match(const Matrix2d& mat) const;
	void set_block_thread_size();

private:
	std::shared_ptr<float> d_data_ = nullptr;
	Size size_;
	int num_elements_;
	dim3 blocks_;
	dim3 threads_;
};

Matrix2d operator-(float scalar, const Matrix2d& mat);

inline Matrix2d operator+(float scalar, const Matrix2d& mat)
{
	return mat.add(scalar);
}

inline Matrix2d operator*(float scalar, const Matrix2d& mat)
{
	return mat.mul(scalar);
}

inline Matrix2d operator/(float scalar, const Matrix2d& mat)
{
	return mat.inv(scalar);
}

Matrix2d eye(Size size);
