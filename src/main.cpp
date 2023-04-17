#include "matrix2d.h"

#include <spdlog/spdlog.h>

#include <array>
#include <string>

int main(void)
{
	spdlog::set_level(spdlog::level::debug);
	spdlog::set_pattern("[%^%l%$] %v");

	spdlog::debug("Testing");

	Matrix2d mat1({0, 1, 2, 3, 4, 5}, {3, 2});
	Matrix2d mat2({6, 7, 8, 9, 10, 11}, {3, 2});
	Matrix2d vec1({1, 2}, {2, 1});

	auto mat3 = mat1 + mat2;
	spdlog::debug("mat3: {}", fmt::join(mat3.get(), ","));
	mat3 += mat1;
	spdlog::debug("mat3: {}", fmt::join(mat3.get(), ","));
	mat3 -= mat1;
	spdlog::debug("mat3: {}", fmt::join(mat3.get(), ","));
	auto mat4 = (mat3 - mat2) * vec1;
	spdlog::debug("mat4: {}={}", fmt::join(mat4.get(), "+"), mat4.sum());
	mat4 += mat4.sum();
	spdlog::debug("mat4: {}", fmt::join(mat4.get(), ","));
	Matrix2d mat5({1, 3}, 2.0F);
	spdlog::debug("mat5: {}", fmt::join(mat5.get(), ","));
	mat5 = mat5 * mat4;
	spdlog::debug("mat5: {}", fmt::join(mat5.get(), ","));

	spdlog::debug("Test complete!");

	return 0;
}
