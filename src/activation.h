#pragma once

#include "functions.h"
#include "matrix2d.h"

/// @brief The activation functions available for fully connected blocks
enum class Activation
{
	kNone,
	kReLU,
	kLeakyReLU,
	kSigmoid,
	kTanh,
};

inline Matrix2d relu(const Matrix2d& mat)
{
	return mat.max(0.0F);
}

Matrix2d relu_derivative(const Matrix2d& mat)
{
	return mat > 0.0F;
}

inline Matrix2d leaky_relu(const Matrix2d& mat, float alpha = 0.01F)
{
	return mat.max(mat * alpha);
}

inline Matrix2d leaky_relu_derivative(const Matrix2d& mat, float alpha = 0.01F)
{
	return (mat > 0.0F) + (mat < 0.0F) * alpha;
}

inline Matrix2d tanh_derivative(const Matrix2d& mat)
{
	auto der = tanh(mat);
	der *= der;
	return 1.0F - der;
}

inline Matrix2d sigmoid_derivative(const Matrix2d& mat)
{
	auto s = sigmoid(mat);
	return s.mul(1.0F - s);
}

inline void activation(const Activation activation, Matrix2d& x)
{
	switch (activation)
	{
		case Activation::kNone: return;
		case Activation::kReLU: x = relu(x); return;
		case Activation::kLeakyReLU: x = leaky_relu(x); return;
		case Activation::kTanh: x = tanh(x); return;
		case Activation::kSigmoid: x = sigmoid(x); return;
	}
	return;
}

inline Matrix2d activation_derivative(const Activation activation, const Matrix2d& x)
{
	switch (activation)
	{
		case Activation::kNone: return x;
		case Activation::kReLU: return relu_derivative(x);
		case Activation::kLeakyReLU: return leaky_relu_derivative(x);
		case Activation::kTanh: return tanh_derivative(x);
		case Activation::kSigmoid: return sigmoid_derivative(x);
	}
	return x;
}
