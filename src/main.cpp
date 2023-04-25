#include "fully_connected.h"
#include "functions.h"
#include "matrix2d.h"
#include "mnist/mnist.h"
#include "optimiser.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <string>

Matrix2d to_mat(const MNISTObject& item)
{
	return Matrix2d(item.image, {static_cast<int>(item.rows), static_cast<int>(item.cols)});
}

Matrix2d make_target(const MNISTObject& item)
{
	std::vector<float> target;
	target.resize(MNISTObject::nlabels, 0.0F);
	target[item.label] = 1.0F;
	return Matrix2d(target, {MNISTObject::nlabels, 1});
}

int main(void)
{
	spdlog::set_level(spdlog::level::debug);
	spdlog::set_pattern("[%^%l%$] %v");

	// Network parameters
	FCConfig config;
	config.layers = {
		{.size = 300, .activation = Activation::kReLU}, {.size = MNISTObject::nlabels, .activation = Activation::kReLU}};

	FullyConnected fc_nn(config, MNISTObject::cols * MNISTObject::rows);

	MNISTDatasetLoader mnist_loader("./mnist");

	auto train_set = mnist_loader.load_train_set();
	if (train_set.empty()) { return 1; }

	auto test_set = mnist_loader.load_test_set();
	if (test_set.empty()) { return 1; }

	// Training Hyperparameters
	constexpr size_t mini_batch_size = 20;
	constexpr size_t num_epoch = 50;
	constexpr float lr = 0.005F;
	OptimiserConfig optimiser_config = {.grad_norm_clip = 0.5F};

	OptimiserManager optimiser(optimiser_config);
	fc_nn.set_optimiser(optimiser);

	std::mt19937 gen(std::random_device{}());

	for (size_t epoch = 0; epoch < num_epoch; ++epoch)
	{
		fc_nn.train();
		optimiser.set_alpha(lr * (1.0F - static_cast<float>(epoch) / static_cast<float>(num_epoch)));
		std::shuffle(train_set.begin(), train_set.end(), gen);
		for (size_t index = 0; index < train_set.size(); index += mini_batch_size)
		{
			std::vector<Matrix2d> grad;
			float mean_loss = 0;
			for (size_t mini_batch = 0; mini_batch < mini_batch_size; ++mini_batch)
			{
				const auto& item = train_set.at(index + mini_batch);
				auto pred = fc_nn.forward(to_mat(item).flatten());
				auto target = make_target(item);
				auto g = loss_squared_derivative(pred, target);
				grad.push_back(g);
				float loss = loss_squared(pred, target);
				mean_loss += loss;
			}
			mean_loss /= mini_batch_size;
			optimiser.optimise(grad);
			if (index % (mini_batch_size * 100) == 0 && index > 0)
			{
				spdlog::info("[{} / {}] [{} / {}] loss: {}", epoch, num_epoch, index, train_set.size(), mean_loss);
			}
		}

		int num_correct = 0;
		fc_nn.train(false);
		for (const auto& item : test_set)
		{
			auto pred = fc_nn.forward(to_mat(item).flatten());
			int pred_label = 0;
			auto pred_vec = pred.get();
			float max = -1.0F;
			for (size_t i = 0; i < pred_vec.size(); ++i)
			{
				if (pred_vec[i] > max)
				{
					max = pred_vec[i];
					pred_label = static_cast<int>(i);
				}
			}
			if (pred_label == item.label) { ++num_correct; }
		}
		float accuracy = 100.0F * static_cast<float>(num_correct) / static_cast<float>(test_set.size());
		spdlog::info("epoch {} complete, accuracy; {}%", epoch, accuracy);
	}

	return 0;
}
