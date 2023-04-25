#pragma once

#include "activation.h"
#include "linear.h"

#include <cstddef>
#include <vector>

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

	void set_optimiser(OptimiserManager& opt_manager)
	{
		std::shared_ptr<Optimiser> optimiser = nullptr;
		for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l)
		{
			optimiser = opt_manager.build_graph(optimiser);
			layers_.at(l).set_optimiser(optimiser);
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
	bool train_ = false;
};
