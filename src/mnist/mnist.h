#pragma once

#include <spdlog/spdlog.h>

#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

struct MNISTObject
{
	std::vector<float> image;
	std::uint8_t label = -1;
	int index = -1;
	static constexpr std::uint32_t rows = 28U;
	static constexpr std::uint32_t cols = 28U;
	static constexpr std::uint8_t nlabels = 10U;
};

class MNISTDatasetLoader
{
public:
	/// @brief Loads the mnist dataset
	/// @param path The path to load the dataset from
	MNISTDatasetLoader(std::filesystem::path path = "");

	std::vector<MNISTObject> load_train_set() const;
	std::vector<MNISTObject> load_test_set() const;

private:
	std::vector<MNISTObject> load_dataset(std::istream& images, std::istream& labels) const;

private:
	std::filesystem::path path_;
};
