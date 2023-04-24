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

	void train(bool enable = true)
	{
		train_ = enable;
		for (auto& layer : layers_) { layer.train(enable); }
	}

	void set_alpha(float alpha)
	{
		alpha_ = alpha;
		for (auto& layer : layers_) { layer.set_alpha(alpha); }
	}

	void set_grad_norm_clip(float clip)
	{
		for (auto& layer : layers_) { layer.set_grad_norm_clip(clip); }
	}

	Matrix2d forward(const Matrix2d& input)
	{
		Matrix2d output = input;
		for (size_t l = 0; l < layers_.size(); ++l) { output = layers_[l](output); }
		return output;
	}

	std::vector<Matrix2d> backprop(std::vector<Matrix2d> grad)
	{
		for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l) { grad = layers_.at(l).backprop(grad); }

		return grad;
	}

private:
	const FCConfig config_;
	std::vector<Linear> layers_;
	int output_size_;
	bool train_ = false;
	float alpha_ = 1.0F;
};

float loss_squared(const Matrix2d& prediction, const Matrix2d& target)
{
	auto error = prediction - target;
	error *= error;
	return 0.5F * error.sum();
}

Matrix2d loss_squared_derivative(const Matrix2d& prediction, const Matrix2d& target)
{
	return prediction - target;
}
