#pragma once

#include "matrix2d.h"

#include <random>

/// @brief The initialisation type to use for
enum class InitType
{
	kConstant,
	kNormal,
};

Matrix2d initialise(Size size, InitType type, float value)
{
	switch (type)
	{
		case InitType::kConstant:
		{
			break;
		}
		case InitType::kNormal:
		{
			static std::mt19937 gen(std::random_device{}());
			std::normal_distribution<float> rnd(0.0F, value);
			Matrix2d mat(size);
			std::vector<float> vec;
			vec.resize(mat.num_elements());
			for (int i = 0; i < mat.num_elements(); ++i) { vec[i] = rnd(gen); }
			mat.set(vec, size);
			return mat;
		}
	}
	return Matrix2d(size, value);
}
