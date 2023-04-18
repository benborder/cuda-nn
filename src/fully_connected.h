#pragma once

#include "activation.h"
#include "linear.h"

#include <cstddef>
#include <vector>

/// @brief Fully connected configuration
struct FCConfig
{
	// Defines each layer in the network.
	std::vector<LinearConfig> layers = {};
};

class FullyConnected
{
public:
	FullyConnected(FCConfig config, const int input_size) : config_(std::move(config))
	{
		output_size_ = input_size;
		for (auto& layer : config_.layers)
		{
			layers_.emplace_back(layer, output_size_);
			output_size_ = layer.size;
		}
	}

	Matrix2d forward(const Matrix2d& input)
	{
		Matrix2d output = input;
		for (size_t l = 0; l < layers_.size(); ++l) { output = layers_[l](output); }
		return output;
	}

private:
	const FCConfig config_;
	std::vector<Linear> layers_;
	int output_size_;
};

float loss_squared(const Matrix2d& prediction, const Matrix2d& target)
{
	auto error = target - prediction;
	error *= error;
	return 0.5F * error.sum();
}
