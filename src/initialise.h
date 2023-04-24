#pragma once

#include "matrix2d.h"

#include <random>

enum class InitType
{
	kConstant,
	kNormal,
	kKaimingNormal,
};

Matrix2d initialise(Size size, InitType type, float value)
{
	static std::mt19937 gen(std::random_device{}());
	switch (type)
	{
		case InitType::kConstant:
		{
			break;
		}
		case InitType::kNormal:
		{
			std::normal_distribution<float> rnd(0.0F, value);
			Matrix2d mat(size);
			std::vector<float> vec;
			vec.resize(mat.num_elements());
			for (int i = 0; i < mat.num_elements(); ++i) { vec[i] = rnd(gen); }
			mat.set(vec, size);
			return mat;
		}
		case InitType::kKaimingNormal:
		{
			std::normal_distribution<float> rnd(0.0F, value / std::sqrt(static_cast<float>(size.x)));
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
