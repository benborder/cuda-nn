#include "fully_connected.h"
#include "functions.h"
#include "matrix2d.h"
#include "mnist/mnist.h"

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

	FCConfig config;
	config.layers = {
		{512, Activation::kReLU, InitType::kNormal, 0.0F},
		{256, Activation::kReLU, InitType::kNormal, 0.0F},
		{64, Activation::kReLU, InitType::kNormal, 0.0F},
		{MNISTObject::nlabels, Activation::kReLU, InitType::kNormal, 0.0F}};

	FullyConnected fc_nn(config, MNISTObject::cols * MNISTObject::rows);

	MNISTDatasetLoader mnist_loader("./mnist");

	auto train_set = mnist_loader.load_train_set();
	if (!train_set.empty()) { spdlog::debug("MNIST loaded!"); }
	size_t mini_batch_size = 10;
	size_t num_epoch = 30;

	std::mt19937 gen(std::random_device{}());

	for (size_t epoch = 0; epoch < num_epoch; ++epoch)
	{
		std::shuffle(train_set.begin(), train_set.end(), gen);
		for (size_t index = 0; index < train_set.size(); index += mini_batch_size)
		{
			float mean_loss = 0;
			for (size_t mini_batch = 0; mini_batch < mini_batch_size; ++mini_batch)
			{
				const auto& item = train_set.at(index + mini_batch);
				auto pred = fc_nn.forward(to_mat(item).flatten());
				auto target = make_target(item);
				float loss = loss_squared(pred, target);
				if (isnanf(loss))
				{
					spdlog::error("pred: {}", fmt::join(pred.get(), ","));
					spdlog::error("targ: {}", fmt::join(target.get(), ","));
				}
				mean_loss += loss;
			}
			mean_loss /= mini_batch_size;
			if (index % (mini_batch_size * 10) == 0) { spdlog::debug("mean_loss: {}", mean_loss); }
		}
		spdlog::info("epoch {} complete", epoch);
	}

	return 0;
}
