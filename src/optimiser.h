#pragma once

#include "matrix2d.h"

#include <stdexcept>
#include <tuple>
#include <vector>

enum class OptimiserType
{
	kSGD,
};

struct OptimiserConfig
{
	OptimiserType type = OptimiserType::kSGD;
	float grad_norm_clip = 0.0F;
};

class OptimiserManager;

class Optimiser
{
	friend class OptimiserManager;

public:
	virtual std::shared_ptr<Optimiser> add_child(const OptimiserConfig& confi) = 0;
	virtual void set_parameters(const Matrix2d& weights, const Matrix2d& bias) = 0;
	virtual void optimise(const std::vector<Matrix2d>& grad_in) = 0;
	virtual void add_forward_result(std::tuple<Matrix2d, Matrix2d>&& forward_result) = 0;
	virtual void set_alpha(float alpha) = 0;
};

class SGD final : public Optimiser
{
public:
	SGD(float grad_norm_clip) : weights_({1}), bias_({1}), grad_norm_clip_(grad_norm_clip) {}

	std::shared_ptr<Optimiser> add_child(const OptimiserConfig& config)
	{
		std::shared_ptr<Optimiser> opt = std::make_shared<SGD>(config.grad_norm_clip);
		children_.emplace_back(opt);
		return opt;
	}

	void set_parameters(const Matrix2d& weights, const Matrix2d& bias) override
	{
		weights_ = weights;
		bias_ = bias;
	}

	void optimise(const std::vector<Matrix2d>& grad_in) override
	{
		if (grad_in.size() != forward_results_.size()) { throw std::runtime_error(""); }

		std::vector<Matrix2d> grad_out;
		{
			Matrix2d weight_grad(weights_.size(), 0.0F);
			Matrix2d bias_grad(bias_.size(), 0.0F);
			for (size_t b = 0; b < forward_results_.size(); ++b)
			{
				const auto& [a, zprime] = forward_results_[b];
				auto grad = grad_in.at(b).mul(zprime);
				bias_grad += grad;
				weight_grad += grad * a.transpose();
				grad_out.push_back(weights_.transpose() * grad);
			}
			float scale = alpha_ / static_cast<float>(forward_results_.size());
			bias_grad *= scale;
			weight_grad *= scale;
			if (grad_norm_clip_ > 0.0F)
			{
				clip_norm(bias_grad, grad_norm_clip_);
				clip_norm(weight_grad, grad_norm_clip_);
			}
			bias_ -= bias_grad;
			weights_ -= weight_grad;
			forward_results_.clear();
		}
		for (auto& child : children_) { child->optimise(grad_out); }
	}

	void add_forward_result(std::tuple<Matrix2d, Matrix2d>&& forward_result) override
	{
		forward_results_.emplace_back(std::move(forward_result));
	}

	void set_alpha(float alpha) override
	{
		alpha_ = alpha;
		for (auto& child : children_) { child->set_alpha(alpha); }
	}

private:
	Matrix2d weights_;
	Matrix2d bias_;
	float alpha_ = 1.0F;
	float grad_norm_clip_;
	std::vector<std::tuple<Matrix2d, Matrix2d>> forward_results_;
	std::vector<std::shared_ptr<Optimiser>> children_;
};

class OptimiserManager
{
public:
	OptimiserManager(OptimiserConfig config) : config_(config) {}

	std::shared_ptr<Optimiser> build_graph(std::shared_ptr<Optimiser>& parent)
	{
		std::shared_ptr<Optimiser> opt;

		if (parent == nullptr)
		{
			switch (config_.type)
			{
				case OptimiserType::kSGD:
				{
					opt = std::make_shared<SGD>(config_.grad_norm_clip);
					break;
				}
			}
			graphs_.push_back(opt);
		}
		else { opt = parent->add_child(config_); }
		return opt;
	}

	void optimise(const std::vector<Matrix2d>& grad_in)
	{
		for (const auto& child : graphs_) { child->optimise(grad_in); }
	}

	void set_alpha(float alpha)
	{
		for (const auto& child : graphs_) { child->set_alpha(alpha); }
	}

private:
	const OptimiserConfig config_;
	std::vector<std::shared_ptr<Optimiser>> graphs_;
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
