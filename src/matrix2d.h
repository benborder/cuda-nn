#pragma once

#include <vector_types.h>

#include <vector>

struct Size
{
	int y;
	int x;
};

class Matrix2d
{
public:
	Matrix2d(const std::vector<float>& data, Size size);
	Matrix2d(Size size, float scalar);
	Matrix2d(const Matrix2d& mat);
	Matrix2d(Matrix2d&& mat);

	~Matrix2d();

	Matrix2d& operator=(const Matrix2d& mat);
	Matrix2d& operator=(Matrix2d&& mat);

	Size size() const;
	int num_elements() const;

	Matrix2d add(const Matrix2d& mat) const;
	Matrix2d sub(const Matrix2d& mat) const;
	Matrix2d mul(const Matrix2d& mat) const;

	Matrix2d add(const float scalar) const;
	Matrix2d sub(const float scalar) const;
	Matrix2d mul(const float scalar) const;

	Matrix2d& operator+=(const Matrix2d& mat);
	Matrix2d operator+(const Matrix2d& mat) const;

	Matrix2d& operator-=(const Matrix2d& mat);
	Matrix2d operator-(const Matrix2d& mat) const;

	Matrix2d& operator*=(const Matrix2d& mat);
	Matrix2d operator*(const Matrix2d& mat) const;

	Matrix2d& operator+=(const float scalar);
	Matrix2d operator+(const float scalar) const;

	Matrix2d& operator-=(const float scalar);
	Matrix2d operator-(const float scalar) const;

	Matrix2d& operator*=(const float scalar);
	Matrix2d operator*(const float scalar) const;

	float sum() const;

	void fill(float scalar);

	float get(int y, int x) const;
	std::vector<float> get() const;

	void set(const std::vector<float>& data, Size size);

private:
	Matrix2d(Size size);
	void check_bounds_match(const Matrix2d& mat) const;

private:
	float* d_data_ = nullptr;
	Size size_;
	int num_elements_;
	dim3 blocks_ = {64, 64, 1};
	dim3 threads_ = {8, 8, 1};
};
