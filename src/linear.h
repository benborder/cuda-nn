#pragma once

#include "initialise.h"
#include "matrix2d.h"

#include <tuple>

struct LinearConfig
{
	// The number of neural units in a layer
	int size = 0;
	// The activation function used for forward passes
	Activation activation = Activation::kNone;
	// The type of initialisation for the weights
	InitType init_weight_type = InitType::kKaimingNormal;
	// The weight values to initialise the network with (if relevant)
	float init_weight = 1.0F;
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

	void train(bool enable = true) { train_ = enable; }

	void set_alpha(float alpha) { alpha_ = alpha; }

	void set_grad_norm_clip(float clip) { grad_norm_clip_ = clip; }

	Matrix2d forward(const Matrix2d& input)
	{
		auto output = weights_ * input + bias_;
		if (train_) { batch_output_.push_back({input, output}); }
		activation(config_.activation, output);
		return output;
	}

	Matrix2d operator()(const Matrix2d& input) { return forward(input); }

	std::vector<Matrix2d> backprop(const std::vector<Matrix2d>& grad_in)
	{
		std::vector<Matrix2d> grad_out;
		Matrix2d weight_grad(weights_.size(), 0.0F);
		Matrix2d bias_grad(bias_.size(), 0.0F);
		for (size_t b = 0; b < batch_output_.size(); ++b)
		{
			const auto& [a, z] = batch_output_[b];
			auto grad = grad_in.at(b).mul(activation_derivative(config_.activation, z));
			bias_grad += grad;
			weight_grad += grad * a.transpose();
			grad_out.push_back(weights_.transpose() * grad);
		}
		// Update weights and bias
		float scale = alpha_ / static_cast<float>(batch_output_.size());
		bias_grad *= scale;
		weight_grad *= scale;
		clip_norm(bias_grad, grad_norm_clip_);
		clip_norm(weight_grad, grad_norm_clip_);
		bias_ -= bias_grad;
		weights_ -= weight_grad;
		batch_output_.clear();
		return grad_out;
	}

private:
	const LinearConfig config_;
	Matrix2d weights_;
	Matrix2d bias_;
	bool train_ = false;
	float alpha_ = 1.0F;
	float grad_norm_clip_ = 0.5F;
	std::vector<std::tuple<Matrix2d, Matrix2d>> batch_output_;
};
