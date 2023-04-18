#pragma once

#include "initialise.h"
#include "matrix2d.h"

struct LinearConfig
{
	// The number of neural units in a layer
	int size = 0;
	// The activation function used for forward passes
	Activation activation = Activation::kNone;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kConstant;
	// The weight values to initialise the network with (if relevant)
	float init_weight = 0.0F;
	// The type of initialisation for the bias
	InitType init_bias_type = InitType::kConstant;
	// The bias values to initialise the network with
	float init_bias = 0.0F;
};

class Linear
{
public:
	Linear(LinearConfig config, int input_size)
			: config_(config)
			, weights_(initialise({config.size, input_size}, config.init_weight_type, config.init_weight))
			, bias_(initialise({config.size, 1}, config.init_bias_type, config.init_bias))
	{
	}

	Matrix2d forward(const Matrix2d& input)
	{
		auto output = weights_ * input + bias_;
		if (train_) { batch_output_.push_back({input, output}); }
		activation(config_.activation, output);
		return output;
	}

	Matrix2d operator()(const Matrix2d& input) { return forward(input); }

private:
	const LinearConfig config_;
	Matrix2d weights_;
	Matrix2d bias_;
};
