#pragma once

#include "matrix2d.h"

inline Matrix2d softmax(const Matrix2d& mat)
{
	auto exp = mat.exp();
	return exp / exp.sum();
}

inline Matrix2d sigmoid(const Matrix2d& mat)
{
	return 1.0F / (1.0F + (-mat).exp());
}

inline Matrix2d tanh(const Matrix2d& mat)
{
	// (e^x – e^-x) / (e^x + e^-x)
	auto ex = mat.exp();
	auto enx = (-mat).exp();
	return (ex - enx) / (ex + enx);
}

inline void clip_norm(Matrix2d& mat, float clip)
{
	float mag = mat.mag();
	if (mag > clip) { mat *= clip / mag; }
}
