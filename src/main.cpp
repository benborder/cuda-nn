#include "matrix2d.h"

#include <spdlog/spdlog.h>

#include <array>
#include <string>

int main(void)
{
	spdlog::set_level(spdlog::level::debug);
	spdlog::set_pattern("[%^%l%$] %v");

	spdlog::debug("Testing");

	Matrix2d mat1({0, 1, 2, 3, 4, 5}, 3, 2);
	Matrix2d mat2({6, 7, 8, 9, 10, 11}, 3, 2);
	Matrix2d vec1({1, 2}, 2);

	auto mat3 = mat1 + mat2;
	mat3 += mat1;
	mat3 -= mat1;
	auto mat4 = (mat3 - mat2) * vec1;

	auto res = mat4.get();

	spdlog::debug("Test complete: {}", fmt::join(res, ","));

	return 0;
}
